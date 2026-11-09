import sys

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


def test_responses(client):
    response = client.responses.create(
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    assert "isinstance" in response.output_text


def test_image_input(client):
    response = client.responses.create(
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        instructions="You are a helpful assistant that describes images.",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Fischotter%2C_Lutra_Lutra.JPG",
                    },
                ],
            }
        ],
    )
    assert 'otter' in response.output_text

def test_stream(client):
    stream = client.responses.create(
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        input=[
            {
                "role": "user",
                "content": "Say 'double bubble bath' ten times fast.",
            },
        ],
        stream=True,
    )

    full_output = ""
    for event in stream:
        if event.type == "response.output_text.delta":
            full_output += event.delta

    assert full_output == '\n'.join(['double bubble bath']*10)
