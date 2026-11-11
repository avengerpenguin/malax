import pytest

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


@pytest.mark.parametrize("limit_param", ["max_tokens", "max_completion_tokens"])
def test_token_limit_params(client, limit_param):
    # Legacy clients send max_tokens; newer SDKs send max_completion_tokens. Both must work.
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a long essay about the history of the Python language."}],
        **{limit_param: 16},
    )
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage.completion_tokens <= 16
