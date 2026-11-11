MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_string_input(client):
    # The shared adapter always sends a message list; this covers the plain-string form.
    response = client.responses.create(
        model=MODEL,
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    assert "isinstance" in response.output_text


def test_incomplete_details(client):
    response = client.responses.create(
        model=MODEL,
        input="Write a long essay about the history of the Python language.",
        max_output_tokens=16,
    )
    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"


def test_echoes_request_params(client):
    response = client.responses.create(
        model=MODEL,
        input="Invent a name for a pet goldfish.",
        temperature=1.5,
        top_p=0.95,
        max_output_tokens=8,
    )
    assert (response.temperature, response.top_p) == (1.5, 0.95)
    assert response.max_output_tokens == 8