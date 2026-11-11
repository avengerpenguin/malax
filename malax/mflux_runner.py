"""Image generation and editing with mflux: model resolution, input checks and generation."""

from __future__ import annotations

import base64
import importlib
import inspect
import io
import json
import math
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fastapi import HTTPException
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from PIL import Image
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    # Type-only: server.py imports this module, so a runtime import would be circular.
    from malax.server import ActiveModel


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


@dataclass(frozen=True)
class MfluxModel:
    """An image model id this runner accepts, and the task it performs."""

    id: str
    task: ImageTask


def models_for_repo(repo_id: str) -> list[MfluxModel]:
    """Image models servable from a Hugging Face repo; empty if mflux can't run it."""
    config = MFLUX_CONFIG_BY_REPO.get(repo_id)
    if config is None:
        return []
    return [
        MfluxModel(id=f"{class_path}/{config}", task=task)
        for task, class_path in mflux_classes(config).items()
    ]


def model_config_names() -> list[str]:
    return sorted(MFLUX_CONFIG_NAMES)


def mflux_loader(request: ImageGenerationRequest) -> tuple[type, Callable[[], Any]]:
    """Resolve '<module>.<Class>/<config or repo>' into the class and a function that builds the model."""
    # `config_ref`, not `config_name`, so the module-level config_name() isn't shadowed.
    class_path, _, config_ref = request.model.partition("/")
    module_path, _, class_name = class_path.rpartition(".")

    if not module_path or not config_ref:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image model {request.model}.Image model must look like '<module>.<Class>/<config or repo>', "
            "e.g. 'flux2.variants.Flux2Klein/flux2_klein_4b'",
        )

    try:
        model_class = getattr(
            importlib.import_module(f"mflux.models.{module_path}"), class_name
        )
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Unknown mflux model class '{class_path}': {e}"
        )

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

    factory = getattr(ModelConfig, config_ref.replace("-", "_"), None)
    if callable(factory):
        kwargs["model_config"] = factory()
    else:
        # A HF repo id or local path: infer the architecture, take the weights from the repo.
        try:
            kwargs["model_config"] = ModelConfig.from_name(
                config_ref, base_model=request.base_model
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Unknown ModelConfig, repo or path '{config_ref}': {e}",
                    "hint": "For third-party repos, set base_model in extra_body, e.g. 'flux2-klein-4b'",
                    "available": model_config_names(),
                },
            )
        kwargs["model_path"] = config_ref

    return model_class, lambda: model_class(**kwargs)


# The mflux CLI's default. The Python API defaults to None, which skips the input image without an error.
DEFAULT_IMAGE_STRENGTH = 0.4


def default_steps(params: dict[str, inspect.Parameter]) -> int | None:
    param = params.get("num_inference_steps")
    return (
        param.default if param is not None and isinstance(param.default, int) else None
    )


def image_inputs(
    request: ImageGenerationRequest, model_class: type, images: list[Image.Image]
) -> dict[str, Any]:
    """Pass in-memory input images the way this class's generate_image takes them; checked before loading."""
    params = inspect.signature(model_class.generate_image).parameters

    if not images:
        if request.image_strength is not None:
            raise HTTPException(
                status_code=400,
                detail="image_strength only applies to /v1/images/edits",
            )
        return {}

    if request.image_strength is not None and "image_strength" not in params:
        raise HTTPException(
            status_code=400, detail=f"'{request.model}' does not take image_strength"
        )

    # mflux's ImageUtil.load_image accepts PIL images as well as paths, despite the parameter names.
    if "image_paths" in params:
        return {"image_paths": images}

    if "image_path" in params:
        if len(images) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"'{request.model}' takes exactly one input image, got {len(images)}",
            )

        strength = (
            request.image_strength
            if request.image_strength is not None
            else DEFAULT_IMAGE_STRENGTH
        )
        if not 0 < strength < 1:
            raise HTTPException(
                status_code=400,
                detail="image_strength must be between 0 and 1 (exclusive)",
            )

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
    active_model: ActiveModel,
    request: ImageGenerationRequest,
    task: ImageTask,
    images: list[Image.Image] | None = None,
) -> dict[str, Any]:
    """Load the requested mflux model and generate images; shared by generations and edits."""
    images = images or []
    request = request.model_copy(
        update={"model": expand_image_model(request.model, task)}
    )
    model_class, loader = mflux_loader(request)
    inputs = image_inputs(request, model_class, images)
    size = request.size_kwargs(images[0] if images else None)
    if images:
        described = {
            k: (f"<{len(v)} images>" if isinstance(v, list) else v)
            for k, v in inputs.items()
        }
        print(
            f"Image inputs for {request.model}: {described}, sizes={[im.size for im in images]}, output={size}"
        )

    try:
        async with active_model.session():
            model = await active_model.load(request.model_key(), loader)
            images_b64 = await active_model.run(
                generate_images_sync, model, request, size, **inputs
            )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "created": int(time.time()),
        "data": [{"b64_json": image} for image in images_b64],
    }
