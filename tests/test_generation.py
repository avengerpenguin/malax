from conftest import data_uri

OTTER = (
    "https://upload.wikimedia.org/wikipedia/commons/d/d3/Fischotter%2C_Lutra_Lutra.JPG"
)
GOLDFISH = "Invent a name for a pet goldfish."


def test_answers_question(api):
    r = api.create(
        system="You are a coding assistant that talks like a pirate.",
        user="How do I check if a Python object is an instance of a class?",
    )
    assert "isinstance" in r.text


def test_stream(api):
    assert api.stream(user="Say 'double bubble bath' ten times fast.") == "\n".join(
        ["double bubble bath"] * 10
    )


def test_truncates_at_token_limit(api):
    r = api.create(
        user="Write a long essay about the history of the Python language.",
        max_tokens=16,
    )
    assert r.truncated


def test_stops_naturally_under_token_limit(api):
    # A reply that ends on its own must not be reported as truncated, even with a tight limit.
    r = api.create(
        user="Reply with only the word 'yes' and nothing else.",
        temperature=0,
        max_tokens=32,
    )
    assert "yes" in r.text.lower(), (
        f"model didn't follow the prompt, so this test proves nothing: {r.text!r}"
    )
    assert not r.truncated


def test_temperature_zero_is_deterministic(api):
    def ask():
        return api.create(user=GOLDFISH, temperature=0, max_tokens=32)

    assert ask().text == ask().text


def test_high_temperature_varies(api):
    def ask():
        return api.create(user=GOLDFISH, temperature=1.5, top_p=0.95, max_tokens=32)

    assert ask().text != ask().text


def test_image_input(api):
    r = api.create(
        system="You are a helpful assistant that describes images.",
        user="What is in this image?",
        image_url=OTTER,
    )
    assert "otter" in r.text


def test_data_uri_image_input(api):
    # The UI's only image path, and it needs no network.
    r = api.create(
        user="What fruit is in this image? Answer in one word.",
        image_url=data_uri("apple.jpg", "image/jpeg"),
        temperature=0,
    )
    assert "apple" in r.text.lower()
