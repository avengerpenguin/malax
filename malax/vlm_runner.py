"""Text generation, with optional image inputs, via mlx-vlm."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import HTTPException
from huggingface_hub.errors import RepositoryNotFoundError
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

from malax.mlx_runtime import ActiveModel


@dataclass
class GenerationStats:
    tokens: int = 0
    prompt_tokens: int = 0
    finish_reason: str | None = None

    def truncated(self, limit: int) -> bool:
        # Prefer the library's own verdict; fall back to comparing against the limit.
        if self.finish_reason is not None:
            return self.finish_reason == "length"
        return self.tokens >= limit


async def stream_mlx_generate(
    *,
    active_model: ActiveModel,
    model_name: str,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    messages: list[dict[str, str]],
    images: list[Image.Image],
    stats: GenerationStats,
) -> AsyncIterator[str]:
    """Stream text deltas; `stats` is filled in as generation goes, and is final once the stream ends."""
    async with active_model.session():
        try:
            model, processor, config = await active_model.load(
                model_name, lambda: (*load(model_name), load_config(model_name))
            )
        except RepositoryNotFoundError:
            # Hugging Face also reports gated or private repos this way (it answers 401 for missing repos too).
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found on Hugging Face or in the local cache",
            )
        prompt = apply_chat_template(
            processor, config, messages, num_images=len(images)
        )

        chunks = stream_generate(
            model,
            processor,
            prompt,
            images or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Each step runs on the single MLX thread; the lock is held until the stream ends.
        while (chunk := await active_model.run(next, chunks, None)) is not None:
            # Not one chunk per token: mlx_vlm yields an extra result after the loop to flush the
            # detokenizer, so trust its own running count instead of counting chunks.
            stats.tokens = getattr(chunk, "generation_tokens", None) or stats.tokens
            stats.prompt_tokens = (
                getattr(chunk, "prompt_tokens", 0) or stats.prompt_tokens
            )
            stats.finish_reason = (
                getattr(chunk, "finish_reason", None) or stats.finish_reason
            )
            if chunk.text:
                yield chunk.text
