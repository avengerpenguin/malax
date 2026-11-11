import asyncio
import base64
import binascii
import io
import itertools
import json
import os
import time
import urllib.parse
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import mlx.core as mx
import requests
from PIL import Image
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import CacheNotFound
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartParser

from malax.mflux_runner import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageTask,
    models_for_repo,
    run_mflux,
)
from malax.mlx_runtime import ActiveModel
from malax.vlm_runner import GenerationStats, stream_mlx_generate

USER_AGENT = os.getenv("HTTP_USER_AGENT", "malax/0.1 (local ML gateway)")
# MLX keeps freed buffers for reuse; uncapped, that cache can be as big as the model itself.
MLX_CACHE_LIMIT_GB = float(os.getenv("MLX_CACHE_LIMIT_GB", "2"))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "50")) * 1024**2)
# Starlette spools uploads over 1MB to a temp file. Images must never touch the disk, so raise the
# threshold to the request cap; images_edits rejects anything bigger before parsing the form.
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES


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


DEFAULT_MAX_TOKENS = 512


class ChatImageURL(BaseModel):
    url: str
    detail: str | None = None


class ChatContentPart(BaseModel):
    type: str
    text: str | None = None
    # The spec uses {"url": ...}; some legacy clients send a bare string.
    image_url: ChatImageURL | str | None = None


class ChatMessage(BaseModel):
    role: str
    # None is valid for assistant messages that only carried tool calls.
    content: str | list[ChatContentPart] | None = None


class ChatStreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # Legacy clients send max_tokens; newer SDKs send max_completion_tokens.
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: ChatStreamOptions | None = None

    def token_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or DEFAULT_MAX_TOKENS


@asynccontextmanager
async def lifespan(app: FastAPI):
    executor = ThreadPoolExecutor(max_workers=1)
    active_model = ActiveModel(executor)
    app.state.mlx_executor = executor
    app.state.active_model = active_model
    mx.set_cache_limit(int(MLX_CACHE_LIMIT_GB * 1024**3))
    print("Local ML gateway starting")

    yield

    await active_model.unload()
    executor.shutdown(wait=True, cancel_futures=False)


app = FastAPI(
    title="MaLaX: Local MLX Gateway",
    lifespan=lifespan,
)


def get_mlx_executor(request: Request) -> ThreadPoolExecutor:
    return request.app.state.mlx_executor


def get_active_model(request: Request) -> ActiveModel:
    return request.app.state.active_model


MlxExecutorDep = Annotated[ThreadPoolExecutor, Depends(get_mlx_executor)]
ActiveModelDep = Annotated[ActiveModel, Depends(get_active_model)]


def now_unix() -> int:
    return int(time.time())


IMAGE_ENDPOINTS: dict[ImageTask, str] = {
    "generate": "/v1/images/generations",
    "edit": "/v1/images/edits",
}


def active_model_id(name: str | None) -> str | None:
    """Image models are keyed by a JSON blob of load-time fields; chat models by repo id."""
    try:
        return json.loads(name)["model"]
    except (TypeError, ValueError, KeyError):
        return name


def list_cached_models_sync(active_name: str | None) -> list[dict[str, Any]]:
    """Model repos in the Hugging Face cache, shaped like OpenAI model objects."""
    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return []

    active = active_model_id(active_name)
    entries = []

    for repo in sorted(cache.repos, key=lambda r: r.repo_id.lower()):
        if repo.repo_type != "model":
            continue

        # Extensions beyond id/object/created/owned_by; the OpenAI SDK ignores unknown fields.
        common = {
            "object": "model",
            "created": int(repo.last_modified),
            "owned_by": repo.repo_id.partition("/")[0]
            if "/" in repo.repo_id
            else "local",
            "repo_id": repo.repo_id,
            "size_on_disk": repo.size_on_disk,
        }

        image_models = models_for_repo(repo.repo_id)

        if not image_models:
            entries.append(
                {"id": repo.repo_id, **common, "active": active == repo.repo_id}
            )
            continue

        for model in image_models:
            entries.append(
                {
                    "id": model.id,
                    **common,
                    "endpoint": IMAGE_ENDPOINTS[model.task],
                    "active": active == model.id,
                }
            )

    return entries


def decode_data_uri(uri: str) -> bytes:
    """Decode a data URI (RFC 2397), e.g. 'data:image/png;base64,iVBOR...'."""
    header, sep, payload = uri[len("data:") :].partition(",")
    if not sep:
        raise HTTPException(status_code=400, detail="Malformed data URI: missing ','")

    if header.split(";")[-1].strip().lower() == "base64":
        try:
            # Clients sometimes wrap long base64 payloads; strip whitespace before strict decoding.
            return base64.b64decode("".join(payload.split()), validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(
                status_code=400, detail="Malformed data URI: invalid base64 payload"
            )

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
        raise HTTPException(
            status_code=400, detail=f"Input image from {source} is not a readable image"
        )
    return image


def to_chat_messages(
    request: ResponsesRequest,
) -> tuple[list[dict[str, str]], list[str]]:
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


def chat_to_messages(
    request: ChatCompletionsRequest,
) -> tuple[list[dict[str, str]], list[str]]:
    """Flatten Chat Completions messages into text-only chat messages plus a list of image URLs."""
    messages = []
    images = []

    for message in request.messages:
        # "developer" is the newer name for "system"; chat templates only know "system".
        role = "system" if message.role == "developer" else message.role

        if message.content is None or isinstance(message.content, str):
            messages.append({"role": role, "content": message.content or ""})
            continue

        texts = []
        for part in message.content:
            if part.type == "text":
                texts.append(part.text or "")
            elif part.type == "image_url" and part.image_url is not None:
                url = (
                    part.image_url
                    if isinstance(part.image_url, str)
                    else part.image_url.url
                )
                images.append(url)
        messages.append({"role": role, "content": "\n".join(texts)})

    return messages, images


async def fetch_images(urls: list[str]) -> list[Image.Image]:
    return list(
        await asyncio.gather(*(asyncio.to_thread(fetch_image, url) for url in urls))
    )


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

    truncated = stats.truncated(request.max_output_tokens)
    yield event(
        "response.incomplete" if truncated else "response.completed",
        response=response_object(**ids, text=text, truncated=truncated),
    )


def chat_usage(stats: GenerationStats) -> dict[str, int]:
    return {
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.tokens,
        "total_tokens": stats.prompt_tokens + stats.tokens,
    }


def chat_finish_reason(request: ChatCompletionsRequest, stats: GenerationStats) -> str:
    return "length" if stats.truncated(request.token_limit()) else "stop"


def chat_completion_object(
    *,
    request: ChatCompletionsRequest,
    completion_id: str,
    text: str,
    stats: GenerationStats,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": now_unix(),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": chat_finish_reason(request, stats),
                "logprobs": None,
            }
        ],
        "usage": chat_usage(stats),
    }


async def stream_chat_chunks(
    *,
    chunks: AsyncIterator[str],
    stats: GenerationStats,
    request: ChatCompletionsRequest,
    completion_id: str,
) -> AsyncIterator[str]:
    created = now_unix()

    def chunk(choices: list[dict[str, Any]], **extra: Any) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": choices,
            **extra,
        }
        return f"data: {json.dumps(payload)}\n\n"

    def choice(
        delta: dict[str, Any], finish_reason: str | None = None
    ) -> dict[str, Any]:
        return {
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
            "logprobs": None,
        }

    yield chunk([choice({"role": "assistant", "content": ""})])

    async for delta in chunks:
        yield chunk([choice({"content": delta})])

    yield chunk([choice({}, chat_finish_reason(request, stats))])

    if request.stream_options and request.stream_options.include_usage:
        yield chunk([], usage=chat_usage(stats))

    yield "data: [DONE]\n\n"


@app.get("/health")
async def health(active_model: ActiveModelDep):
    return {
        "status": "ok",
        "model": {
            "active": active_model.name,
            "busy": active_model.lock.locked(),
            "waiting_requests": active_model.waiting_requests,
        },
    }


@app.get("/v1/models")
async def list_models(active_model: ActiveModelDep):
    # Scanning the cache walks the filesystem, so keep it off the event loop.
    return {
        "object": "list",
        "data": await asyncio.to_thread(list_cached_models_sync, active_model.name),
    }


@app.get("/v1/models/{model:path}")
async def retrieve_model(model: str, active_model: ActiveModelDep):
    # `:path` because HF repo ids contain a slash, e.g. "mlx-community/Qwen2.5-VL-7B-Instruct-4bit".
    for entry in await asyncio.to_thread(list_cached_models_sync, active_model.name):
        if entry["id"] == model:
            return entry
    raise HTTPException(
        status_code=404, detail=f"Model '{model}' not found in the Hugging Face cache"
    )


@app.post("/v1/responses")
async def responses(request: ResponsesRequest, active_model: ActiveModelDep):
    messages, image_urls = to_chat_messages(request)
    images = await fetch_images(image_urls)

    stats = GenerationStats()
    chunks = stream_mlx_generate(
        active_model=active_model,
        model_name=request.model,
        max_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        messages=messages,
        images=images,
        stats=stats,
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
    truncated = stats.truncated(request.max_output_tokens)
    return response_object(**ids, text=text, truncated=truncated)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionsRequest, active_model: ActiveModelDep
):
    messages, image_urls = chat_to_messages(request)
    images = await fetch_images(image_urls)

    stats = GenerationStats()
    chunks = stream_mlx_generate(
        active_model=active_model,
        model_name=request.model,
        max_tokens=request.token_limit(),
        temperature=request.temperature,
        top_p=request.top_p,
        messages=messages,
        images=images,
        stats=stats,
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if request.stream:
        return StreamingResponse(
            stream_chat_chunks(
                chunks=chunks, stats=stats, request=request, completion_id=completion_id
            ),
            media_type="text/event-stream",
        )

    text = "".join([chunk async for chunk in chunks])
    return chat_completion_object(
        request=request, completion_id=completion_id, text=text, stats=stats
    )


@app.post("/v1/images/generations")
async def images_generations(
    request: ImageGenerationRequest, active_model: ActiveModelDep
):
    return await run_mflux(active_model, request, "generate")


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
async def images_edits(http_request: Request, active_model: ActiveModelDep):
    # Checked before parsing so no upload can exceed the in-memory spool and spill to disk.
    length = http_request.headers.get("content-length")
    if length is None:
        raise HTTPException(status_code=411, detail="Content-Length is required")
    if int(length) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Request exceeds {MAX_UPLOAD_BYTES // 1024**2}MB"
        )

    form = await http_request.form()

    # The OpenAI SDK sends a list of images as `image[]` and a single image as `image`.
    uploads = form.getlist("image[]") or form.getlist("image")
    if not uploads or not all(isinstance(u, UploadFile) for u in uploads):
        raise HTTPException(
            status_code=400, detail="At least one image file is required"
        )
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
        # e.errors() can hold the raw exception from a model_validator in `ctx`; e.json() stringifies it.
        raise HTTPException(
            status_code=422, detail=json.loads(e.json(include_url=False))
        )

    images = [
        decode_upload(i, upload.filename, await upload.read())
        for i, upload in enumerate(uploads)
    ]
    return await run_mflux(active_model, request, "edit", images=images)


INDEX_HTML = Path(__file__).parent / "index.html"


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(INDEX_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5276, workers=1, timeout_keep_alive=3600)
