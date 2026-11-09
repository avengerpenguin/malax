from openai import OpenAI
from fastapi.testclient import TestClient
import pytest


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


def test_chat_completions(client):
    response = client.chat.completions.create(
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        messages=[{"role": "user", "content": "What is 6 times 7?"}],
    )
    assert "42" in response.choices[0].message.content


def test_responses(client):
    response = client.responses.create(
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    assert "isinstance" in response.output_text
