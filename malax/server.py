import asyncio
import base64
import binascii
import gc
import importlib
import inspect
import io
import itertools
import json
import math
import os
import random
import time
import traceback
import urllib.parse
import uuid
import weakref
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import CacheNotFound
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartParser


USER_AGENT = os.getenv("HTTP_USER_AGENT", "malax/0.1 (local ML gateway)")
# MLX keeps freed buffers for reuse; uncapped, that cache can be as big as the model itself.
MLX_CACHE_LIMIT_GB = float(os.getenv("MLX_CACHE_LIMIT_GB", "2"))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "50")) * 1024**2)
# Starlette spools uploads over 1MB to a temp file. Images must never touch the disk, so raise the
# threshold to the request cap; images_edits rejects anything bigger before parsing the form.
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES
mlx_executor: ThreadPoolExecutor | None = None


async def run_on_mlx_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(mlx_executor, lambda: func(*args, **kwargs))


def clear_mlx_cache_sync():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def load_sync(loader: Callable[[], Any]) -> Any:
    model = loader()
    # Free intermediates from loading (e.g. LoRA baking) before generation needs the memory.
    gc.collect()
    mx.clear_cache()
    return model


def memory_stats() -> str:
    gb = 1024**3
    return (
        f"active={mx.get_active_memory() / gb:.1f}GB "
        f"cache={mx.get_cache_memory() / gb:.1f}GB "
        f"peak={mx.get_peak_memory() / gb:.1f}GB"
    )


class ActiveModel:
    """Holds at most one model (chat or image) in memory at a time."""

    def __init__(self):
        self.name: str | None = None
        self.model: Any | None = None
        self.lock = asyncio.Lock()
        self.waiting_requests: int = 0

    @asynccontextmanager
    async def session(self):
        self.waiting_requests += 1
        try:
            await self.lock.acquire()
        finally:
            self.waiting_requests -= 1

        try:
            yield
        finally:
            self.lock.release()

    async def load(self, name: str, loader: Callable[[], Any]) -> Any:
        if self.name == name:
            return self.model

        await self.unload()

        print(f"Loading model: {name}")
        self.model = await run_on_mlx_thread(load_sync, loader)
        self.name = name
        print(f"Loaded model: {name} ({memory_stats()})")

        return self.model

    async def unload(self):
        if self.model is None:
            return

        print(f"Unloading model: {self.name} ({memory_stats()})")

        # Chat models are a (model, processor, config) tuple; track the model itself.
        target = self.model[0] if isinstance(self.model, tuple) else self.model
        try:
            ref = weakref.ref(target)
        except TypeError:
            ref = None
        del target

        self.model = None
        self.name = None

        await run_on_mlx_thread(clear_mlx_cache_sync)

        if ref is not None and (leaked := ref()) is not None:
            holders = [type(r).__name__ for r in gc.get_referrers(leaked)]
            print(f"WARNING: model still alive after unload, referenced by: {holders}")
            del leaked
        print(f"Unloaded ({memory_stats()})")


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


SIZE_PATTERN = r"^(auto|\d+x\d+)$"

# Request fields baked in when the model is constructed; changing any of them means a reload.
LOAD_TIME_FIELDS = {"model", "quantize", "base_model", "lora_paths", "lora_scales"}


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int = Field(1, ge=1, le=10)
    size: str = Field("1024x1024", pattern=SIZE_PATTERN)
    response_format: Literal["b64_json"] = "b64_json"

    # mflux extensions; send these via `extra_body` from the OpenAI SDK.
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    negative_prompt: str | None = None
    image_strength: float | None = None
    quantize: int | None = None
    base_model: str | None = None
    lora_paths: list[str] = []
    lora_scales: list[float] = []

    @model_validator(mode="after")
    def check_lora_scales(self):
        if self.lora_scales and len(self.lora_scales) != len(self.lora_paths):
            raise ValueError("lora_scales must have one entry per lora_paths entry")
        return self

    def model_key(self) -> str:
        return json.dumps(self.model_dump(include=LOAD_TIME_FIELDS))

    def size_kwargs(self, reference: Image.Image | None = None) -> dict[str, int]:
        """For "auto", follow the reference image's aspect ratio at up to ~1MP, so mflux never reads it from a path."""
        if self.size != "auto":
            width, height = map(int, self.size.split("x"))
            return {"width": width, "height": height}
        if reference is None:
            return {}
        width, height = reference.size
        scale = min(1.0, math.sqrt(1024 * 1024 / (width * height)))
        return {
            "width": max(16, int(width * scale) // 16 * 16),
            "height": max(16, int(height * scale) // 16 * 16),
        }


class ImageEditRequest(ImageGenerationRequest):
    size: str = Field("auto", pattern=SIZE_PATTERN)


ImageTask = Literal["generate", "edit"]

IMAGE_ENDPOINTS: dict[ImageTask, str] = {
    "generate": "/v1/images/generations",
    "edit": "/v1/images/edits",
}

# mflux has no mapping from a ModelConfig to the class that runs it, so map config-name
# prefixes by hand. The longest prefix wins; add a row when you start using a new family.
MFLUX_CLASSES: dict[str, dict[ImageTask, str]] = {
    "flux2_klein": {
        "generate": "flux2.variants.Flux2Klein",
        "edit": "flux2.variants.Flux2KleinEdit",
    },
    "z_image": {"generate": "z_image.ZImage"},
}


def config_name(alias: str) -> str:
    """Prefer the ModelConfig factory spelling (flux2_klein_4b), as used in the tests."""
    name = alias.replace("-", "_")
    return name if callable(getattr(ModelConfig, name, None)) else alias


MFLUX_CONFIG_NAMES = {config_name(alias) for alias in AVAILABLE_MODELS}

def config_preference(config: str) -> tuple[bool, int]:
    """Sort key: ModelConfig factory spellings first, then the shortest alias."""
    return (not callable(getattr(ModelConfig, config, None)), len(config))


# HF repo id -> config name, e.g. "black-forest-labs/FLUX.2-klein-4B" -> "flux2_klein_4b".
# Several aliases can share a repo (e.g. Tongyi-MAI/Z-Image-Turbo), so pick the best one.
MFLUX_CONFIG_BY_REPO: dict[str, str] = {}
for _alias, _config in AVAILABLE_MODELS.items():
    _candidate = config_name(_alias)
    _current = MFLUX_CONFIG_BY_REPO.get(_config.model_name)
    if _current is None or config_preference(_candidate) < config_preference(_current):
        MFLUX_CONFIG_BY_REPO[_config.model_name] = _candidate


def mflux_classes(config: str) -> dict[ImageTask, str]:
    normalized = config.replace("-", "_")
    matches = [prefix for prefix in MFLUX_CLASSES if normalized.startswith(prefix)]
    return MFLUX_CLASSES[max(matches, key=len)] if matches else {}


def expand_image_model(name: str, task: ImageTask) -> str:
    """Turn a bare config name or repo id into '<module>.<Class>/<config>'; leave anything else alone."""
    config = config_name(MFLUX_CONFIG_BY_REPO.get(name, name))
    if config not in MFLUX_CONFIG_NAMES:
        return name
    class_path = mflux_classes(config).get(task)
    return f"{class_path}/{config}" if class_path else name


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mlx_executor

    mlx_executor = ThreadPoolExecutor(max_workers=1)
    mx.set_cache_limit(int(MLX_CACHE_LIMIT_GB * 1024**3))
    print("Local ML gateway starting")

    yield

    await active_model.unload()
    mlx_executor.shutdown(wait=True, cancel_futures=False)


app = FastAPI(
    title="Local ML Gateway",
    lifespan=lifespan,
)


def model_config_names() -> list[str]:
    return sorted(MFLUX_CONFIG_NAMES)


def mflux_loader(request: ImageGenerationRequest) -> tuple[type, Callable[[], Any]]:
    """Resolve '<module>.<Class>/<config or repo>' into the class and a function that builds the model."""
    class_path, _, config_name = request.model.partition("/")
    module_path, _, class_name = class_path.rpartition(".")

    if not module_path or not config_name:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image model {request.model}.Image model must look like '<module>.<Class>/<config or repo>', "
            "e.g. 'flux2.variants.Flux2Klein/flux2_klein_4b'",
        )

    try:
        model_class = getattr(importlib.import_module(f"mflux.models.{module_path}"), class_name)
    except (ImportError, AttributeError) as e:
        raise HTTPException(status_code=400, detail=f"Unknown mflux model class '{class_path}': {e}")

    # Only pass what the client set, so mflux applies its own defaults.
    kwargs: dict[str, Any] = {
        name: value
        for name, value in {
            "quantize": request.quantize,
            "lora_paths": request.lora_paths,
            "lora_scales": request.lora_scales,
        }.items()
        if value
    }

    factory = getattr(ModelConfig, config_name.replace("-", "_"), None)
    if callable(factory):
        kwargs["model_config"] = factory()
    else:
        # A HF repo id or local path: infer the architecture, take the weights from the repo.
        try:
            kwargs["model_config"] = ModelConfig.from_name(config_name, base_model=request.base_model)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Unknown ModelConfig, repo or path '{config_name}': {e}",
                    "hint": "For third-party repos, set base_model in extra_body, e.g. 'flux2-klein-4b'",
                    "available": model_config_names(),
                },
            )
        kwargs["model_path"] = config_name

    return model_class, lambda: model_class(**kwargs)


# The mflux CLI's default. The Python API defaults to None, which skips the input image without an error.
DEFAULT_IMAGE_STRENGTH = 0.4


def default_steps(params: dict[str, inspect.Parameter]) -> int | None:
    param = params.get("num_inference_steps")
    return param.default if param is not None and isinstance(param.default, int) else None


def image_inputs(request: ImageGenerationRequest, model_class: type, images: list[Image.Image]) -> dict[str, Any]:
    """Pass in-memory input images the way this class's generate_image takes them; checked before loading."""
    params = inspect.signature(model_class.generate_image).parameters

    if not images:
        if request.image_strength is not None:
            raise HTTPException(status_code=400, detail="image_strength only applies to /v1/images/edits")
        return {}

    if request.image_strength is not None and "image_strength" not in params:
        raise HTTPException(status_code=400, detail=f"'{request.model}' does not take image_strength")

    # mflux's ImageUtil.load_image accepts PIL images as well as paths, despite the parameter names.
    if "image_paths" in params:
        return {"image_paths": images}

    if "image_path" in params:
        if len(images) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"'{request.model}' takes exactly one input image, got {len(images)}",
            )

        strength = request.image_strength if request.image_strength is not None else DEFAULT_IMAGE_STRENGTH
        if not 0 < strength < 1:
            raise HTTPException(status_code=400, detail="image_strength must be between 0 and 1 (exclusive)")

        # img2img skips the first int(steps * strength) denoising steps; at 0 the image is ignored.
        steps = request.steps if request.steps is not None else default_steps(params)
        if steps is not None and int(steps * strength) < 1:
            raise HTTPException(
                status_code=400,
                detail=f"steps={steps} with image_strength={strength} would ignore the input image; "
                "raise one of them",
            )

        return {"image_path": images[0], "image_strength": strength}

    raise HTTPException(
        status_code=400,
        detail=f"'{request.model}' does not accept input images; use /v1/images/generations",
    )


def png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def generate_images_sync(
    model: Any, request: ImageGenerationRequest, size: dict[str, int], **inputs: Any
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
                **size,
                **inputs,
                **options,
            ).image
        )
        for i in range(request.n)
    ]


async def run_mflux(
    request: ImageGenerationRequest, task: ImageTask, images: list[Image.Image] | None = None
) -> dict[str, Any]:
    """Load the requested mflux model and generate images; shared by generations and edits."""
    images = images or []
    request = request.model_copy(update={"model": expand_image_model(request.model, task)})
    model_class, loader = mflux_loader(request)
    inputs = image_inputs(request, model_class, images)
    size = request.size_kwargs(images[0] if images else None)
    if images:
        described = {k: (f"<{len(v)} images>" if isinstance(v, list) else v) for k, v in inputs.items()}
        print(f"Image inputs for {request.model}: {described}, sizes={[im.size for im in images]}, output={size}")

    try:
        async with active_model.session():
            model = await active_model.load(request.model_key(), loader)
            images_b64 = await run_on_mlx_thread(generate_images_sync, model, request, size, **inputs)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "created": now_unix(),
        "data": [{"b64_json": image} for image in images_b64],
    }


def now_unix() -> int:
    return int(time.time())


def active_model_id() -> str | None:
    """Image models are keyed by a JSON blob of load-time fields; chat models by repo id."""
    try:
        return json.loads(active_model.name)["model"]
    except (TypeError, ValueError, KeyError):
        return active_model.name


def list_cached_models_sync() -> list[dict[str, Any]]:
    """Model repos in the Hugging Face cache, shaped like OpenAI model objects."""
    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return []

    active = active_model_id()
    entries = []

    for repo in sorted(cache.repos, key=lambda r: r.repo_id.lower()):
        if repo.repo_type != "model":
            continue

        # Extensions beyond id/object/created/owned_by; the OpenAI SDK ignores unknown fields.
        common = {
            "object": "model",
            "created": int(repo.last_modified),
            "owned_by": repo.repo_id.partition("/")[0] if "/" in repo.repo_id else "local",
            "repo_id": repo.repo_id,
            "size_on_disk": repo.size_on_disk,
        }

        config = MFLUX_CONFIG_BY_REPO.get(repo.repo_id)
        classes = mflux_classes(config) if config else {}

        if not classes:
            entries.append({"id": repo.repo_id, **common, "active": active == repo.repo_id})
            continue

        for task, class_path in classes.items():
            model_id = f"{class_path}/{config}"
            entries.append(
                {"id": model_id, **common, "endpoint": IMAGE_ENDPOINTS[task], "active": active == model_id}
            )

    return entries


def decode_data_uri(uri: str) -> bytes:
    """Decode a data URI (RFC 2397), e.g. 'data:image/png;base64,iVBOR...'."""
    header, sep, payload = uri[len("data:"):].partition(",")
    if not sep:
        raise HTTPException(status_code=400, detail="Malformed data URI: missing ','")

    if header.split(";")[-1].strip().lower() == "base64":
        try:
            # Clients sometimes wrap long base64 payloads; strip whitespace before strict decoding.
            return base64.b64decode("".join(payload.split()), validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Malformed data URI: invalid base64 payload")

    return urllib.parse.unquote_to_bytes(payload)


def fetch_image(url: str) -> Image.Image:
    """Load an input image from an http(s) URL or a base64 data URI, fully decoded in memory."""
    if url.startswith("data:"):
        data = decode_data_uri(url)
        source = "data URI"
    else:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        data = response.content
        source = url

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Input image from {source} is not a readable image")
    return image


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
            "waiting_requests": active_model.waiting_requests,
        },
    }


@app.get("/v1/models")
async def list_models():
    # Scanning the cache walks the filesystem, so keep it off the event loop.
    return {"object": "list", "data": await asyncio.to_thread(list_cached_models_sync)}


@app.get("/v1/models/{model:path}")
async def retrieve_model(model: str):
    # `:path` because HF repo ids contain a slash, e.g. "mlx-community/Qwen2.5-VL-7B-Instruct-4bit".
    for entry in await asyncio.to_thread(list_cached_models_sync):
        if entry["id"] == model:
            return entry
    raise HTTPException(status_code=404, detail=f"Model '{model}' not found in the Hugging Face cache")


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
    return await run_mflux(request, "generate")


def decode_upload(index: int, filename: str | None, data: bytes) -> Image.Image:
    """Fully decode in memory, so truncated or non-image uploads fail here rather than mid-generation."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Upload {index} ('{filename}') is empty or not a readable image",
        )
    return image


@app.post("/v1/images/edits")
async def images_edits(http_request: Request):
    # Checked before parsing so no upload can exceed the in-memory spool and spill to disk.
    length = http_request.headers.get("content-length")
    if length is None:
        raise HTTPException(status_code=411, detail="Content-Length is required")
    if int(length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Request exceeds {MAX_UPLOAD_BYTES // 1024**2}MB")

    form = await http_request.form()

    # The OpenAI SDK sends a list of images as `image[]` and a single image as `image`.
    uploads = form.getlist("image[]") or form.getlist("image")
    if not uploads or not all(isinstance(u, UploadFile) for u in uploads):
        raise HTTPException(status_code=400, detail="At least one image file is required")
    if "mask" in form:
        raise HTTPException(status_code=400, detail="mask is not supported")

    # Multipart has no arrays; the SDK sends lists as repeated `key[]` fields.
    fields: dict[str, Any] = {}
    for key in form.keys():
        if key in ("image", "image[]"):
            continue
        if key.endswith("[]"):
            fields[key[:-2]] = form.getlist(key)
        else:
            fields[key] = form[key]

    try:
        request = ImageEditRequest(**fields)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    images = [
        decode_upload(i, upload.filename, await upload.read())
        for i, upload in enumerate(uploads)
    ]
    return await run_mflux(request, "edit", images=images)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5276, workers=1, timeout_keep_alive=3600)
