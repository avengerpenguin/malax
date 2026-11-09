import asyncio
import gc
import io
import itertools
import json
import os
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pydantic import BaseModel
from mflux.models.common.config import ModelConfig
from mflux.models.krea2 import Krea2
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.z_image import ZImage
from mflux.models.flux3.variants import Flux3Klein


SUPPORTED_IMAGE_MODELS = {
    "z-image": {
        "backend": "mflux",
        "model_class": ZImage,
        "model_config_factory": lambda: ModelConfig.z_image_turbo(),
        "quantize": int(os.getenv("Z_IMAGE_MFLUX_QUANTIZE", "8")),
        "default_width": 768,
        "default_height": 1024,
        "default_steps": 3,
        "default_guidance": None,
        "supports_negative_prompt": True,
    },
    "krea2": {
        "backend": "mflux",
        "model_class": Krea2,
        "model_config_factory": lambda: ModelConfig.krea2(),
        "quantize": int(os.getenv("KREA2_MFLUX_QUANTIZE", "4")),
        "default_width": 768,
        "default_height": 1024,
        "default_steps": 8,
        "default_guidance": 1.5,
        "supports_negative_prompt": True,
    },
    "flux3": {
        "backend": "mflux",
        "model_class": Flux3Klein,
        "model_config_factory": lambda: ModelConfig.flux3_image_klein(),
        "quantize": int(os.getenv("FLUX3_MFLUX_QUANTIZE", "4")),
        "default_width": 768,
        "default_height": 1024,
        "default_steps": 16,
        "default_guidance": 1.5,
        "supports_negative_prompt": False,
    },
    "qwen": {
        "backend": "mflux",
        "model_class": QwenImage,
        "model_config_factory": lambda: ModelConfig.qwen_image(),
        "quantize": int(os.getenv("QWEN_IMAGE_MFLUX_QUANTIZE", "4")),
        "default_width": 768,
        "default_height": 1024,
        "default_steps": 4,
        "default_guidance": 4.0,
        "supports_negative_prompt": True,
    },
    "qwen21": {
        "backend": "mflux",
        "model_class": QwenImage21,
        "model_config_factory": lambda: ModelConfig.qwen_image_21(),
        "quantize": int(os.getenv("QWEN21_MFLUX_QUANTIZE", "4")),
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 20,
        "default_guidance": 4.5,
        "supports_negative_prompt": True,
    },
}

DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "sdxl")

DEFAULT_NEGATIVE_PROMPT = (
    "3d render, cg, digital art, illustration, painting, anime, deformed, mutated, bad anatomy, "
    "bad hands, missing fingers, extra limbs, blurry, smooth skin, plastic skin, oversaturated, "
    "airbrushed, drawing, watermark, text"
)

USER_AGENT = os.getenv("HTTP_USER_AGENT", "malax/0.1 (local ML gateway)")

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
    model: str = DEFAULT_IMAGE_MODEL
    prompt: str
    negative_prompt: str = ""
    steps: int = 25
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: int | None = 42


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mlx_executor

    mlx_executor = ThreadPoolExecutor(max_workers=1)

    print("Local ML gateway starting")
    print(f"Supported image models: {list(SUPPORTED_IMAGE_MODELS.keys())}")
    print(f"Default image model: {DEFAULT_IMAGE_MODEL}")

    yield

    await active_model.unload()
    mlx_executor.shutdown(wait=True, cancel_futures=False)


app = FastAPI(
    title="Local ML Gateway",
    lifespan=lifespan,
)


def resolve_image_model_name(requested_model: str | None) -> str:
    model_name = requested_model or DEFAULT_IMAGE_MODEL
    model_name = model_name.lower()

    aliases = {
        "zimage": "z-image",
        "z-image-turbo": "z-image",
        "krea": "krea2",
        "krea-2": "krea2",
        "qwen": "qwen",
        "qwen-image": "qwen",
        "qwen-image-2.1": "qwen21",
        "qwen2.1": "qwen21",
        "qwen-2.1": "qwen21",
    }
    model_name = aliases.get(model_name, model_name)

    if model_name not in SUPPORTED_IMAGE_MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported image model: {model_name}",
                "supported_models": list(SUPPORTED_IMAGE_MODELS.keys()),
            },
        )

    return model_name


def image_to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    for attr_name in ("image", "pil_image", "pil", "_image"):
        nested_image = getattr(image, attr_name, None)
        if isinstance(nested_image, Image.Image):
            return nested_image

    raise TypeError(f"Unsupported image result type: {type(image)!r}")


def image_to_png_stream(image: Any) -> io.BytesIO:
    pil_image = image_to_pil_image(image)

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


def load_mflux_model_sync(model_name: str) -> Any:
    config = SUPPORTED_IMAGE_MODELS[model_name]
    return config["model_class"](
        quantize=config["quantize"],
        model_config=config["model_config_factory"](),
    )


def run_mflux_image_generation_sync(
    model: Any,
    model_name: str,
    request: ImageGenerationRequest,
):
    config = SUPPORTED_IMAGE_MODELS[model_name]

    width = request.width if request.width != 1024 else config["default_width"]
    height = request.height if request.height != 1024 else config["default_height"]
    steps = request.steps if request.steps != 25 else config["default_steps"]

    kwargs = {
        "seed": request.seed if request.seed is not None else 42,
        "prompt": request.prompt,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
    }

    default_guidance = config.get("default_guidance")
    if default_guidance is not None:
        kwargs["guidance"] = (
            request.guidance_scale
            if request.guidance_scale != 7.5
            else default_guidance
        )

    if config.get("supports_negative_prompt"):
        kwargs["negative_prompt"] = request.negative_prompt or DEFAULT_NEGATIVE_PROMPT

    return model.generate_image(**kwargs)


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
        "image": {
            "default_model": DEFAULT_IMAGE_MODEL,
            "supported_models": list(SUPPORTED_IMAGE_MODELS.keys()),
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


@app.post("/image/generate", response_class=StreamingResponse)
async def image_generate(request: ImageGenerationRequest):
    model_name = resolve_image_model_name(request.model)

    try:
        async with active_model.session():
            model = await active_model.load(
                model_name, lambda: load_mflux_model_sync(model_name)
            )
            image = await run_on_mlx_thread(
                run_mflux_image_generation_sync,
                model,
                model_name,
                request,
            )
    except Exception as e:
        print("\n=== ERROR GENERATING IMAGE ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(image_to_png_stream(image), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5276, workers=1, timeout_keep_alive=3600)
