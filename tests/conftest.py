import pytest
from openai import OpenAI
from fastapi.testclient import TestClient


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

