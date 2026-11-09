import base64
from pathlib import Path

HERE = Path(__file__).parent


def test_image_create(client):
    prompt = """
    A children's book drawing of a veterinarian using a stethoscope to
    listen to the heartbeat of a baby otter.
    """

    result = client.images.generate(
        model="flux2.variants.Flux2Klein/flux2_klein_4b",
        prompt=prompt,
        extra_body={"seed": 42, "steps": 4},
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open(HERE / "otter.png", "wb") as f:
        f.write(image_bytes)


def test_image_edit(client):
    prompt = """
    An oil painting of one of these vases standing next to
    the fresh red apple on a dark, rustic wooden table.
    """

    with open(HERE / "apple.jpg", "rb") as apple, open(HERE / "vases.jpg", "rb") as vases:
        result = client.images.edit(
            model="flux2.variants.Flux2KleinEdit/flux2_klein_4b",
            image=[apple, vases],
            prompt=prompt,
            extra_body={"seed": 42},
        )

    image_bytes = base64.b64decode(result.data[0].b64_json)

    with open(HERE / "painting.png", "wb") as f:
        f.write(image_bytes)