import pytest
from openai import OpenAI
from fastapi.testclient import TestClient

from dataclasses import dataclass

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


@pytest.fixture
def server():
    from malax.server import app
    return app


@pytest.fixture
def http_client(server):
    with TestClient(server) as c:
        yield c


@pytest.fixture
def client(http_client):
    return OpenAI(
        api_key="dummy",
        base_url="http://testserver/v1",
        http_client=http_client,
    )

@dataclass
class Result:
    text: str
    truncated: bool


class ResponsesAPI:
    def __init__(self, client):
        self.client = client

    def _content(self, user, image_url):
        if image_url is None:
            return user
        return [
            {"type": "input_text", "text": user},
            {"type": "input_image", "image_url": image_url},
        ]

    def create(self, *, user, system=None, image_url=None, max_tokens=None, **sampling) -> Result:
        kwargs = {"max_output_tokens": max_tokens} if max_tokens else {}
        r = self.client.responses.create(
            model=MODEL,
            instructions=system,
            input=[{"role": "user", "content": self._content(user, image_url)}],
            **kwargs,
            **sampling,
        )
        return Result(text=r.output_text, truncated=r.status == "incomplete")

    def stream(self, *, user) -> str:
        events = self.client.responses.create(
            model=MODEL, input=[{"role": "user", "content": user}], stream=True
        )
        return "".join(e.delta for e in events if e.type == "response.output_text.delta")


class ChatCompletionsAPI:
    def __init__(self, client):
        self.client = client

    def _messages(self, user, system, image_url):
        content = user if image_url is None else [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        system_msg = [{"role": "system", "content": system}] if system else []
        return [*system_msg, {"role": "user", "content": content}]

    def create(self, *, user, system=None, image_url=None, max_tokens=None, **sampling) -> Result:
        kwargs = {"max_tokens": max_tokens} if max_tokens else {}
        c = self.client.chat.completions.create(
            model=MODEL, messages=self._messages(user, system, image_url), **kwargs, **sampling
        )
        choice = c.choices[0]
        return Result(text=choice.message.content, truncated=choice.finish_reason == "length")

    def stream(self, *, user) -> str:
        chunks = self.client.chat.completions.create(
            model=MODEL, messages=self._messages(user, None, None), stream=True
        )
        return "".join(
            ch.choices[0].delta.content
            for ch in chunks
            if ch.choices and ch.choices[0].delta.content
        )


@pytest.fixture(params=[ResponsesAPI, ChatCompletionsAPI], ids=["responses", "chat"])
def api(request, client):
    return request.param(client)

