import pytest

from conftest import MODEL

pytestmark = pytest.mark.slow


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


def test_multi_turn_history(client):
    response = client.responses.create(
        model=MODEL,
        temperature=0,
        input=[
            {"role": "user", "content": "My name is Zebulon. Please remember it."},
            {
                "role": "assistant",
                "content": "Nice to meet you, Zebulon. I'll remember.",
            },
            {"role": "user", "content": "What is my name? Reply with just the name."},
        ],
    )
    assert "zebulon" in response.output_text.lower()


@pytest.mark.parametrize(
    "prompt, limit, final",
    [
        (
            "Write a long essay about the history of the Python language.",
            8,
            "response.incomplete",
        ),
        ("Reply with only the word 'yes'.", 256, "response.completed"),
    ],
    ids=["truncated", "natural-stop"],
)
def test_stream_event_sequence(client, prompt, limit, final):
    events = list(
        client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=limit,
            temperature=0,
            stream=True,
        )
    )
    types = [e.type for e in events]
    assert types[0] == "response.created" and types[-1] == final, (
        f"text: {events[-1].response.output_text!r}"
    )
    assert set(types[1:-1]) == {"response.output_text.delta"}
    assert [e.sequence_number for e in events] == list(range(len(events)))
    # The final event carries the full text the deltas added up to.
    assert events[-1].response.output_text == "".join(e.delta for e in events[1:-1])
