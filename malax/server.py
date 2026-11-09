import asyncio
import gc
import io
import json
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Literal

import mlx.core as mx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mlx_lm import generate, load
from PIL import Image
from pydantic import BaseModel, ConfigDict
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

text_mlx_executor = ThreadPoolExecutor(max_workers=1)
image_mlx_executor = ThreadPoolExecutor(max_workers=1)


async def run_on_executor(executor: ThreadPoolExecutor, func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: func(*args, **kwargs),
    )


async def run_on_text_mlx_thread(func, *args, **kwargs):
    return await run_on_executor(text_mlx_executor, func, *args, **kwargs)


async def run_on_image_mlx_thread(func, *args, **kwargs):
    return await run_on_executor(image_mlx_executor, func, *args, **kwargs)


def clear_mlx_cache_sync():
    gc.collect()

    try:
        mx.clear_cache()
    except AttributeError:
        try:
            mx.metal.clear_cache()
        except AttributeError:
            pass


def clear_torch_cache_sync():
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


class ActiveMlxModel:
    def __init__(self):
        self.model_name: str | None = None
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.lock = asyncio.Lock()
        self.active_request_id: str | None = None
        self.waiting_requests: int = 0

    def is_loaded(self, model_name: str) -> bool:
        return (
            self.model_name == model_name
            and self.model is not None
            and self.tokenizer is not None
        )

    async def acquire_for_request(self, request_id: str):
        self.waiting_requests += 1
        try:
            await self.lock.acquire()
        finally:
            self.waiting_requests -= 1

        self.active_request_id = request_id

    def release_request(self):
        self.active_request_id = None
        self.lock.release()

    async def load_for_request(self, model_name: str):
        if self.is_loaded(model_name):
            return self.model, self.tokenizer

        await self.unload()

        print(f"Loading MLX model: {model_name}")
        self.model, self.tokenizer = await run_on_text_mlx_thread(load, model_name)
        self.model_name = model_name
        print(f"Loaded MLX model: {model_name}")

        return self.model, self.tokenizer

    async def unload(self):
        if self.model is None and self.tokenizer is None:
            return

        print(f"Unloading MLX model: {self.model_name}")

        self.model = None
        self.tokenizer = None
        self.model_name = None

        await run_on_text_mlx_thread(clear_mlx_cache_sync)


active_mlx_model = ActiveMlxModel()


class ActiveImagePipeline:
    def __init__(self):
        self.model_name: str | None = None
        self.pipeline: Any | None = None
        self.lock = asyncio.Lock()
        self.active_request_id: str | None = None
        self.waiting_requests: int = 0

    def is_loaded(self, model_name: str) -> bool:
        return self.model_name == model_name and self.pipeline is not None

    async def acquire_for_request(self, request_id: str):
        self.waiting_requests += 1
        try:
            await self.lock.acquire()
        finally:
            self.waiting_requests -= 1

        self.active_request_id = request_id

    def release_request(self):
        self.active_request_id = None
        self.lock.release()

    async def load_for_request(self, model_name: str):
        if self.is_loaded(model_name):
            return self.pipeline

        await self.unload()

        config = SUPPORTED_IMAGE_MODELS[model_name]

        if config.get("backend") != "diffusers":
            raise RuntimeError(f"Image model {model_name} is not a Diffusers model")

        print(f"Loading image model: {model_name} ({config['model_id']})")

        load_kwargs = {
            "torch_dtype": config["torch_dtype"],
        }

        if config["variant"] is not None:
            load_kwargs["variant"] = config["variant"]

        pipe = await asyncio.to_thread(
            config["pipeline"].from_pretrained,
            config["model_id"],
            **load_kwargs,
        )

        if torch.backends.mps.is_available():
            pipe = pipe.to("mps")

        self.pipeline = pipe
        self.model_name = model_name

        print(f"Loaded image model: {model_name}")

        return self.pipeline

    async def unload(self):
        if self.pipeline is None:
            return

        print(f"Unloading image model: {self.model_name}")

        self.pipeline = None
        self.model_name = None

        await asyncio.to_thread(clear_torch_cache_sync)


active_image_pipeline = ActiveImagePipeline()


class ActiveMfluxImageModel:
    def __init__(self):
        self.model_name: str | None = None
        self.model: Any | None = None

    def is_loaded(self, model_name: str) -> bool:
        return self.model_name == model_name and self.model is not None

    async def load_for_request(self, model_name: str):
        if self.is_loaded(model_name):
            return self.model

        await self.unload()

        config = SUPPORTED_IMAGE_MODELS[model_name]
        model_class = config["model_class"]

        if model_class is None or ModelConfig is None:
            raise RuntimeError(
                f"mflux model class for {model_name} is unavailable. Is mflux installed?"
            )

        print(
            f"Loading mflux image model: {model_name} quantize={config.get('quantize')}"
        )

        constructor_kwargs = {
            "quantize": config.get("quantize"),
        }

        model_config_factory = config.get("model_config_factory")
        if model_config_factory is not None:
            constructor_kwargs["model_config"] = model_config_factory()

        self.model = await run_on_image_mlx_thread(model_class, **constructor_kwargs)
        self.model_name = model_name

        print(f"Loaded mflux image model: {model_name}")

        return self.model

    async def unload(self):
        if self.model is None:
            return

        print(f"Unloading mflux image model: {self.model_name}")

        self.model = None
        self.model_name = None

        await run_on_image_mlx_thread(clear_mlx_cache_sync)


active_mflux_image_model = ActiveMfluxImageModel()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512
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
    print("Local ML gateway starting")
    print(f"Supported image models: {list(SUPPORTED_IMAGE_MODELS.keys())}")
    print(f"Default image model: {DEFAULT_IMAGE_MODEL}")

    yield

    await active_mlx_model.unload()
    await active_image_pipeline.unload()
    await active_mflux_image_model.unload()

    text_mlx_executor.shutdown(wait=True, cancel_futures=False)
    image_mlx_executor.shutdown(wait=True, cancel_futures=False)


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


def run_image_generation_sync(
    pipe: Any, model_name: str, request: ImageGenerationRequest
):
    negative_prompt = request.negative_prompt or DEFAULT_NEGATIVE_PROMPT

    if model_name == "qwen":
        width = request.width if request.width != 1024 else 512
        height = request.height if request.height != 1024 else 768
        steps = request.steps if request.steps != 25 else 10

        kwargs = {
            "prompt": request.prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "true_cfg_scale": 1.2,
        }
        if request.seed is not None:
            kwargs["generator"] = torch.Generator("cpu").manual_seed(request.seed)

        output = pipe(**kwargs)
        return output.images[0]

    output = pipe(
        prompt=request.prompt,
        negative_prompt=negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=1.1,
    )

    return output.images[0]


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


def chat_content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    text_parts: list[str] = []

    for part in content:
        if not isinstance(part, dict):
            continue

        if part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    return "\n".join(text_parts)


def chat_messages_to_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str:
    raw_messages = [
        {
            "role": "system" if message.role == "developer" else message.role,
            "content": chat_content_to_text(message.content),
        }
        for message in messages
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            raw_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        "\n".join(
            f"{message.role}: {chat_content_to_text(message.content)}"
            for message in messages
        )
        + "\nassistant:"
    )


def openai_chat_response(
    *,
    model: str,
    content: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now_unix(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        },
    }


def openai_completion_response(*, model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": now_unix(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def run_mlx_chat_generate(
    *,
    model_name: str,
    messages: list[ChatMessage],
    max_tokens: int,
    temperature: float,
) -> str:
    request_id = uuid.uuid4().hex

    await active_mlx_model.acquire_for_request(request_id)
    try:
        model, tokenizer = await active_mlx_model.load_for_request(model_name)
        prompt = chat_messages_to_prompt(tokenizer, messages)

        return await run_on_text_mlx_thread(
            generate,
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            # temp=temperature,
            verbose=False,
        )
    finally:
        active_mlx_model.release_request()


async def run_mlx_completion_generate(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    request_id = uuid.uuid4().hex

    await active_mlx_model.acquire_for_request(request_id)
    try:
        model, tokenizer = await active_mlx_model.load_for_request(model_name)

        return await run_on_text_mlx_thread(
            generate,
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
    finally:
        active_mlx_model.release_request()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mlx": {
            "active_model": active_mlx_model.model_name,
            "busy": active_mlx_model.lock.locked(),
            "active_request_id": active_mlx_model.active_request_id,
            "waiting_requests": active_mlx_model.waiting_requests,
        },
        "image": {
            "default_model": DEFAULT_IMAGE_MODEL,
            "supported_models": list(SUPPORTED_IMAGE_MODELS.keys()),
            "active_model": (
                active_image_pipeline.model_name or active_mflux_image_model.model_name
            ),
            "active_backend": (
                "diffusers"
                if active_image_pipeline.model_name is not None
                else "mflux"
                if active_mflux_image_model.model_name is not None
                else None
            ),
            "busy": active_image_pipeline.lock.locked(),
            "active_request_id": active_image_pipeline.active_request_id,
            "waiting_requests": active_image_pipeline.waiting_requests,
        },
    }



@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model_name = request.model

    if request.stream:
        return StreamingResponse(
            stream_chat_completion_chunks(
                model_name=model_name,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ),
            media_type="text/event-stream",
        )

    content = await run_mlx_chat_generate(
        model_name=model_name,
        messages=request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return openai_chat_response(
        model=model_name,
        content=content,
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    model_name = request.model

    if request.stream:
        return StreamingResponse(
            stream_completion_chunks(
                model_name=model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ),
            media_type="text/event-stream",
        )

    text = await run_mlx_completion_generate(
        model_name=model_name,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return openai_completion_response(
        model=model_name,
        text=text,
    )


async def stream_chat_completion_chunks(
    *,
    model_name: str,
    messages: list[ChatMessage],
    max_tokens: int,
    temperature: float,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = now_unix()

    content = await run_mlx_chat_generate(
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": content,
                },
                "finish_reason": None,
            }
        ],
    }

    yield f"data: {json.dumps(payload)}\n\n"

    done_payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    yield f"data: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_completion_chunks(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
):
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = now_unix()

    text = await run_mlx_completion_generate(
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    payload = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": None,
            }
        ],
    }

    yield f"data: {json.dumps(payload)}\n\n"

    done_payload = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "stop",
            }
        ],
    }

    yield f"data: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/image/generate", response_class=StreamingResponse)
async def image_generate(request: ImageGenerationRequest):
    model_name = resolve_image_model_name(request.model)
    request_id = uuid.uuid4().hex

    await active_image_pipeline.acquire_for_request(request_id)

    try:
        if SUPPORTED_IMAGE_MODELS[model_name].get("backend") == "mflux":
            await active_image_pipeline.unload()

            model = await active_mflux_image_model.load_for_request(model_name)

            image = await run_on_image_mlx_thread(
                run_mflux_image_generation_sync,
                model,
                model_name,
                request,
            )

            return StreamingResponse(
                image_to_png_stream(image),
                media_type="image/png",
            )

        await active_mflux_image_model.unload()

        pipe = await active_image_pipeline.load_for_request(model_name)

        image = await asyncio.to_thread(
            run_image_generation_sync,
            pipe,
            model_name,
            request,
        )

        return StreamingResponse(
            image_to_png_stream(image),
            media_type="image/png",
        )

    except Exception as e:
        print("\n=== ERROR GENERATING IMAGE ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        active_image_pipeline.release_request()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5276, workers=1, timeout_keep_alive=3600)
