import asyncio
import base64
import gc
import importlib
import io
import itertools
import json
import os
import random
import re
import tempfile
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from mflux.models.common.config import ModelConfig
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile


USER_AGENT = os.getenv("HTTP_USER_AGENT", "malax/0.1 (local ML gateway)")

EXAMPLE_IMAGE_MODEL = "flux2.variants.Flux2Klein/flux2_klein_base_4b"

mlx_executor: ThreadPoolExecutor | None = None


async def run_on_mlx_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(mlx_executor, lambda: func(*args, **kwargs))


def clear_mlx_cache_sync():
    gc.collect()

    try:
        mx.clear_cache()
    except AttributeError:
        try:
            mx.metal.clear_cache()
        except AttributeError:
            pass


class ActiveModel:
    """Holds at most one model (chat or image) in memory at a time."""

    def __init__(self):
        self.name: str | None = None
        self.model: Any | None = None
        self.lock = asyncio.Lock()
        self.active_request_id: str | None = None
        self.waiting_requests: int = 0

    @asynccontextmanager
    async def session(self):
        self.waiting_requests += 1
        try:
            await self.lock.acquire()
        finally:
            self.waiting_requests -= 1

        self.active_request_id = uuid.uuid4().hex
        try:
            yield
        finally:
            self.active_request_id = None
            self.lock.release()

    async def load(self, name: str, loader: Callable[[], Any]) -> Any:
        if self.name == name and self.model is not None:
            return self.model

        await self.unload()

        print(f"Loading model: {name}")
        self.model = await run_on_mlx_thread(loader)
        self.name = name
        print(f"Loaded model: {name}")

        return self.model

    async def unload(self):
        if self.model is None:
            return

        print(f"Unloading model: {self.name}")

        self.model = None
        self.name = None

        await run_on_mlx_thread(clear_mlx_cache_sync)


active_model = ActiveModel()


class InputContent(BaseModel):
    type: str
    text: str | None = None
    image_url: str | None = None


class InputMessage(BaseModel):
    role: str
    content: str | list[InputContent]


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[InputMessage]
    instructions: str | None = None
    max_output_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int = Field(1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: str = "b64_json"

    # mflux extensions; send these via `extra_body` from the OpenAI SDK.
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    negative_prompt: str | None = None
    quantize: int | None = None


class ImageEditRequest(ImageGenerationRequest):
    # "auto" lets mflux size the output from the input images.
    size: str = "auto"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mlx_executor

    mlx_executor = ThreadPoolExecutor(max_workers=1)
    print("Local ML gateway starting")

    yield

    await active_model.unload()
    mlx_executor.shutdown(wait=True, cancel_futures=False)


app = FastAPI(
    title="Local ML Gateway",
    lifespan=lifespan,
)


def model_config_names() -> list[str]:
    return sorted(
        name
        for name, attr in vars(ModelConfig).items()
        if isinstance(attr, (staticmethod, classmethod)) and not name.startswith("_")
    )


def resolve_mflux_model(spec: str) -> tuple[type, Callable[[], ModelConfig]]:
    """Resolve '<module>.<Class>/<config>' to an mflux model class and ModelConfig factory."""
    class_path, slash, config_name = spec.partition("/")
    module_path, _, class_name = class_path.rpartition(".")

    if not slash or not module_path or not config_name:
        raise HTTPException(
            status_code=400,
            detail=f"Image model must look like '<module>.<Class>/<config>', "
            f"e.g. '{EXAMPLE_IMAGE_MODEL}'",
        )

    try:
        module = importlib.import_module(f"mflux.models.{module_path}")
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Unknown mflux model class '{class_path}': {e}"
        )

    factory = getattr(ModelConfig, config_name.replace("-", "_"), None)
    if not callable(factory):
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unknown ModelConfig '{config_name}'",
                "available": model_config_names(),
            },
        )

    return model_class, factory


def parse_size(size: str) -> tuple[int, int]:
    if size == "auto":
        return 1024, 1024

    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match:
        raise HTTPException(
            status_code=400, detail=f"Invalid size '{size}', expected WIDTHxHEIGHT"
        )

    return int(match[1]), int(match[2])


def png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def generate_images_sync(
    model: Any, request: ImageGenerationRequest, **inputs: Any
) -> list[str]:
    seed = request.seed if request.seed is not None else random.randrange(2**32)

    # Only pass what the client set, so mflux applies its own per-model defaults.
    options = {
        name: value
        for name, value in {
            "num_inference_steps": request.steps,
            "guidance": request.guidance,
            "negative_prompt": request.negative_prompt,
        }.items()
        if value is not None
    }

    return [
        png_base64(
            model.generate_image(
                seed=seed + i,
                prompt=request.prompt,
                **inputs,
                **options,
            ).image
        )
        for i in range(request.n)
    ]


async def run_mflux(request: ImageGenerationRequest, **inputs: Any) -> dict[str, Any]:
    """Load the requested mflux model and generate images; shared by generations and edits."""
    if request.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="Only response_format=b64_json is supported")

    model_class, config_factory = resolve_mflux_model(request.model)

    # Quantization changes the loaded weights, so it is part of the model identity.
    model_key = f"{request.model}?quantize={request.quantize}"

    try:
        async with active_model.session():
            model = await active_model.load(
                model_key,
                lambda: model_class(quantize=request.quantize, model_config=config_factory()),
            )
            images = await run_on_mlx_thread(generate_images_sync, model, request, **inputs)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "created": now_unix(),
        "data": [{"b64_json": image} for image in images],
    }


def now_unix() -> int:
    return int(time.time())


def fetch_image(url: str) -> Image.Image:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def to_chat_messages(request: ResponsesRequest) -> tuple[list[dict[str, str]], list[str]]:
    """Flatten Responses input into text-only chat messages plus a list of image URLs."""
    items = (
        [InputMessage(role="user", content=request.input)]
        if isinstance(request.input, str)
        else request.input
    )

    messages = []
    images = []

    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    for item in items:
        if isinstance(item.content, str):
            messages.append({"role": item.role, "content": item.content})
            continue

        texts = []
        for part in item.content:
            if part.type == "input_text":
                texts.append(part.text)
            elif part.type == "input_image":
                images.append(part.image_url)
        messages.append({"role": item.role, "content": "\n".join(texts)})

    return messages, images


@dataclass
class GenerationStats:
    tokens: int = 0


async def stream_mlx_generate(
    *,
    request: ResponsesRequest,
    messages: list[dict[str, str]],
    images: list[Image.Image],
    stats: GenerationStats,
) -> AsyncIterator[str]:
    async with active_model.session():
        model, processor, config = await active_model.load(
            request.model, lambda: (*load(request.model), load_config(request.model))
        )
        prompt = apply_chat_template(processor, config, messages, num_images=len(images))

        chunks = stream_generate(
            model,
            processor,
            prompt,
            images or None,
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        # Each step runs on the single MLX thread; the lock is held until the stream ends.
        while (chunk := await run_on_mlx_thread(next, chunks, None)) is not None:
            stats.tokens += 1
            if chunk.text:
                yield chunk.text


def response_object(
    *,
    request: ResponsesRequest,
    response_id: str,
    message_id: str,
    text: str | None,
    truncated: bool = False,
) -> dict[str, Any]:
    if text is None:
        status = "in_progress"
    elif truncated:
        status = "incomplete"
    else:
        status = "completed"

    output = []
    if text is not None:
        output.append(
            {
                "type": "message",
                "id": message_id,
                "status": status,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )

    return {
        "id": response_id,
        "object": "response",
        "created_at": now_unix(),
        "status": status,
        "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
        "model": request.model,
        "output": output,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }


async def stream_response_events(
    *,
    chunks: AsyncIterator[str],
    stats: GenerationStats,
    request: ResponsesRequest,
    response_id: str,
    message_id: str,
) -> AsyncIterator[str]:
    sequence = itertools.count()

    def event(type_: str, **fields: Any) -> str:
        payload = {"type": type_, "sequence_number": next(sequence), **fields}
        return f"event: {type_}\ndata: {json.dumps(payload)}\n\n"

    ids = {"request": request, "response_id": response_id, "message_id": message_id}

    yield event("response.created", response=response_object(**ids, text=None))

    text = ""
    async for delta in chunks:
        text += delta
        yield event(
            "response.output_text.delta",
            item_id=message_id,
            output_index=0,
            content_index=0,
            delta=delta,
            logprobs=[],
        )

    truncated = stats.tokens >= request.max_output_tokens
    yield event(
        "response.incomplete" if truncated else "response.completed",
        response=response_object(**ids, text=text, truncated=truncated),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": {
            "active": active_model.name,
            "busy": active_model.lock.locked(),
            "active_request_id": active_model.active_request_id,
            "waiting_requests": active_model.waiting_requests,
        },
    }


@app.post("/v1/responses")
async def responses(request: ResponsesRequest):
    messages, image_urls = to_chat_messages(request)
    images = await asyncio.gather(
        *(asyncio.to_thread(fetch_image, url) for url in image_urls)
    )

    stats = GenerationStats()
    chunks = stream_mlx_generate(
        request=request, messages=messages, images=images, stats=stats
    )
    ids = {
        "request": request,
        "response_id": f"resp_{uuid.uuid4().hex}",
        "message_id": f"msg_{uuid.uuid4().hex}",
    }

    if request.stream:
        return StreamingResponse(
            stream_response_events(chunks=chunks, stats=stats, **ids),
            media_type="text/event-stream",
        )

    text = "".join([chunk async for chunk in chunks])
    truncated = stats.tokens >= request.max_output_tokens
    return response_object(**ids, text=text, truncated=truncated)


@app.post("/v1/images/generations")
async def images_generations(request: ImageGenerationRequest):
    width, height = parse_size(request.size)
    return await run_mflux(request, width=width, height=height)


@app.post("/v1/images/edits")
async def images_edits(http_request: Request):
    form = await http_request.form()

    # The OpenAI SDK sends a list of images as `image[]` and a single image as `image`.
    uploads = form.getlist("image[]") or form.getlist("image")
    if not uploads or not all(isinstance(u, UploadFile) for u in uploads):
        raise HTTPException(status_code=400, detail="At least one image file is required")
    if "mask" in form:
        raise HTTPException(status_code=400, detail="mask is not supported")

    fields = {k: v for k, v in form.items() if not isinstance(v, UploadFile)}
    try:
        request = ImageEditRequest(**fields)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    inputs = {}
    if request.size != "auto":
        inputs["width"], inputs["height"] = parse_size(request.size)

    # mflux edit models take file paths, so stage the uploads on disk for the request.
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, upload in enumerate(uploads):
            path = Path(tmp) / f"{i}{Path(upload.filename or '').suffix or '.png'}"
            path.write_bytes(await upload.read())
            paths.append(str(path))

        return await run_mflux(request, image_paths=paths, **inputs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5276, workers=1, timeout_keep_alive=3600)
