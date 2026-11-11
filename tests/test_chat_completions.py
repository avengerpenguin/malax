import openai
import pytest

from conftest import MODEL

pytestmark = pytest.mark.slow

ESSAY = [
    {
        "role": "user",
        "content": "Write a long essay about the history of the Python language.",
    }
]


@pytest.mark.parametrize("limit_param", ["max_tokens", "max_completion_tokens"])
def test_token_limit_params(client, limit_param):
    # Legacy clients send max_tokens; newer SDKs send max_completion_tokens. Both must work.
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Write a long essay about the history of the Python language.",
            }
        ],
        **{limit_param: 16},
    )
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage.completion_tokens <= 16


def test_multi_turn_history(client):
    # The UI resends the whole conversation; earlier turns must reach the chat template.
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": "My name is Zebulon. Please remember it."},
            {
                "role": "assistant",
                "content": "Nice to meet you, Zebulon. I'll remember.",
            },
            {"role": "user", "content": "What is my name? Reply with just the name."},
        ],
    )
    assert "zebulon" in completion.choices[0].message.content.lower()


def test_developer_role_acts_as_system(client):
    completion = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "developer",
                "content": "Whatever the user says, reply only with the word 'banana'.",
            },
            {"role": "user", "content": "What colour is the sky?"},
        ],
    )
    assert "banana" in completion.choices[0].message.content.lower()


def test_stream_reports_finish_reason_and_usage(client):
    chunks = list(
        client.chat.completions.create(
            model=MODEL,
            messages=ESSAY,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert chunks[0].choices[0].delta.role == "assistant"
    assert [
        c.choices[0].finish_reason
        for c in chunks
        if c.choices and c.choices[0].finish_reason
    ] == ["length"]
    assert chunks[-1].choices == [] and 0 < chunks[-1].usage.completion_tokens <= 16


def test_unknown_model_is_a_client_error(client):
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(
            model="mlx-community/does-not-exist-malax", messages=ESSAY
        )
