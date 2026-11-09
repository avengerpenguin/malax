import os
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.messages.utils import convert_to_openai_messages, message_chunk_to_message
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr
from textutils import stream_model_to_stdout, strip_thinking_prefix

try:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
except ImportError:
    raise ImportError("Please install karl[mlx] to use MLX proxy server")

DEFAULT_MLX_CHAT_MODEL = os.getenv(
    "MLX_DEFAULT_MODEL",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
)

# Inline reasoning delimiters used by local models that do not expose a separate
# reasoning channel. Classification happens here, once, so consumers never see tags.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _prefix_overlap(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
    for size in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class MLXChatModel(BaseChatModel):
    """Minimal LangChain adapter for in-process mlx-lm models.

    Only `_stream` is implemented; `_generate` is composed from it, so there is a
    single code path. Reasoning is emitted as standard `reasoning` content blocks,
    which means `message.text` yields the answer without the thoughts.
    """

    model_id: str = DEFAULT_MLX_CHAT_MODEL
    max_tokens: int = 512
    temperature: float = 0.85          # Change default to allow narrative variance
    repetition_penalty: float = 1.15   # Inject a healthy repetition penalty
    repetition_context_size: int = 64  # How far back the penalty looks

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model, self._tokenizer = load(
            self.model_id,
            tokenizer_config={"fix_mistral_regex": True},
        )

    @property
    def _llm_type(self) -> str:
        return "mlx-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools],
            tool_choice=tool_choice,
            **kwargs,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # One implementation, two entry points.
        return generate_from_stream(
            self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._to_prompt(messages, tools=kwargs.pop("tools", None))
        kwargs.pop("tool_choice", None)

        # 1. Pull dynamic settings or fall back to class defaults
        temp = kwargs.pop("temperature", self.temperature)
        rep_penalty = kwargs.pop("repetition_penalty", 1.15) # Default a penalty to kill loops
        rep_context = kwargs.pop("repetition_context_size", 64)

        # 2. Build the decoupled MLX components
        sampler = make_sampler(temp=temp)
        logits_processors = make_logits_processors(
            repetition_penalty=rep_penalty,
            repetition_context_size=rep_context
        )

        in_reasoning = prompt.rfind(_THINK_OPEN) > prompt.rfind(_THINK_CLOSE)

        def tokens() -> Iterator[str]:
            emitted = ""

            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                sampler=sampler,                           # Pass clean sampler
                logits_processors=logits_processors,       # Pass penalty list
            ):
                token = getattr(response, "text", "") or ""

                if not token:
                    continue

                start = len(emitted)
                emitted += token

                if stop and (cut := self._stop_index(emitted, stop)) is not None:
                    if head := token[: cut - start]:
                        yield head
                    return

                yield token

        for text, is_reasoning in self._classify(tokens(), in_reasoning):
            yield self._chunk(text, is_reasoning, run_manager)

    @staticmethod
    def _classify(
        tokens: Iterator[str],
        in_reasoning: bool,
    ) -> Iterator[tuple[str, bool]]:
        """Split a token stream into (text, is_reasoning) runs.

        Matching is done against a buffer rather than individual tokens, so a
        delimiter spanning several tokens is still detected, and a token that
        straddles a delimiter is split rather than landing wholly in one channel.
        Output concatenates back to the input exactly.
        """
        pending = ""

        for token in tokens:
            pending += token

            while True:
                tag = _THINK_CLOSE if in_reasoning else _THINK_OPEN
                found = pending.find(tag)

                if found == -1:
                    break

                if in_reasoning:
                    # Thoughts plus the closing tag; the tag is never answer text.
                    yield pending[: found + len(tag)], True
                else:
                    if found:
                        yield pending[:found], False
                    yield tag, True

                pending = pending[found + len(tag) :]
                in_reasoning = not in_reasoning

            # Hold back any tail that could be the start of a delimiter.
            tag = _THINK_CLOSE if in_reasoning else _THINK_OPEN
            keep = _prefix_overlap(pending, tag)

            if len(pending) > keep:
                yield pending[: len(pending) - keep], in_reasoning
                pending = pending[len(pending) - keep :]

        if pending:
            yield pending, in_reasoning

    @staticmethod
    def _chunk(
        token: str,
        is_reasoning: bool,
        run_manager: CallbackManagerForLLMRun | None,
    ) -> ChatGenerationChunk:
        if is_reasoning:
            block: dict[str, Any] = {"type": "reasoning", "reasoning": token, "index": 0}
        else:
            block = {"type": "text", "text": token, "index": 1}
            if run_manager is not None:
                run_manager.on_llm_new_token(token)

        return ChatGenerationChunk(message=AIMessageChunk(content=[block]))

    @staticmethod
    def _stop_index(text: str, stop: list[str]) -> int | None:
        hits = [i for i in (text.find(s) for s in stop) if i != -1]
        return min(hits) if hits else None

    def _to_prompt(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        # Let the model's own chat template render tools in its native syntax.
        return self._tokenizer.apply_chat_template(
            convert_to_openai_messages(messages),
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )


def stream_to_stderr(runnable: Runnable, messages: Input, **kwargs) -> BaseMessage:
    """Mirror everything to stderr; return the merged final message."""
    final: AIMessageChunk | None = None

    for event in runnable.stream(messages, **kwargs):
        # Bare models yield chunks; agents with stream_mode="messages" yield
        # (chunk, metadata) tuples.
        chunk = event[0] if isinstance(event, tuple) else event

        _echo(chunk)

        if isinstance(chunk, AIMessageChunk):
            final = chunk if final is None else final + chunk

    sys.stderr.write("\n")
    sys.stderr.flush()

    if final is None:
        # Nothing was streamed; hand back an empty message so callers can rely
        # on `.text`, `.tool_calls` etc. without a None check.
        return AIMessage(content="")

    # Collapse the chunk into a plain AIMessage so callers can safely append it
    # to a history or serialise it.
    return message_chunk_to_message(final)


def _echo(message: BaseMessage) -> None:
    if message.type == "tool":
        sys.stderr.write(f"\n[tool result] {message.name}: {message.text}\n")
        sys.stderr.flush()
        return

    for block in message.content_blocks:
        kind = block.get("type")

        if kind == "text":
            sys.stderr.write(block.get("text", ""))
        elif kind == "reasoning":
            sys.stderr.write(block.get("reasoning", ""))
        elif kind in ("tool_call", "tool_call_chunk"):
            name = block.get("name") or ""
            args = block.get("args", "")
            sys.stderr.write(f"\n[tool call] {name} {args}\n")

        sys.stderr.flush()
