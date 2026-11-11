import base64
from pathlib import Path
import openai
import pytest
from PIL import Image

from malax.mflux_runner import ImageEditRequest

pytestmark = pytest.mark.slow

SEED = 1729
SIZE = "512x512"

HERE = Path(__file__).parent
OUTPUT = HERE / "output"

OTTER_PROMPT = """
A children's book drawing of a veterinarian using a stethoscope to
listen to the heartbeat of a baby otter.
"""


def save(result, name: str, size: tuple[int, int] = (512, 512)) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / name
    path.write_bytes(base64.b64decode(result.data[0].b64_json))
    with Image.open(path) as image:
        assert image.format == "PNG" and image.size == size
    return path


def test_image_create(client):
    result = client.images.generate(
        model="flux2.variants.Flux2Klein/flux2_klein_4b",
        prompt=OTTER_PROMPT,
        size=SIZE,
        extra_body={"seed": SEED, "steps": 4},
    )
    save(result, "otter_flux.png")


def test_image_create_zimage(client):
    result = client.images.generate(
        model="z_image.ZImage/z_image_turbo",
        prompt=OTTER_PROMPT,
        size=SIZE,
        extra_body={
            "seed": SEED,
            "steps": 6,
            "quantize": 8,
            "lora_paths": ["renderartist/Technically-Color-Z-Image-Turbo"],
            "lora_scales": [0.5],
        },
    )
    save(result, "otter_zimage.png")


def test_image_edit(client):
    prompt = """
    An oil painting of one of these vases standing next to
    the fresh red apple on a dark, rustic wooden table.
    """

    with (
        open(HERE / "apple.jpg", "rb") as apple,
        open(HERE / "vases.jpg", "rb") as vases,
    ):
        result = client.images.edit(
            model="flux2.variants.Flux2KleinEdit/flux2_klein_4b",
            image=[apple, vases],
            prompt=prompt,
            size=SIZE,
            extra_body={"seed": SEED, "steps": 4},
        )
    save(result, "painting.png")


def test_image_edit_zimage(client):
    with open(HERE / "car.png", "rb") as car:
        result = client.images.edit(
            model="z_image.ZImage/z_image_turbo",
            image=[car],
            prompt="A watercolour painting of a car",
            size=SIZE,
            extra_body={"seed": SEED, "steps": 6, "image_strength": 0.5, "quantize": 8},
        )
    save(result, "car_watercolour.png")


def test_image_edit_rejects_non_image(client):
    with pytest.raises(openai.BadRequestError):
        client.images.edit(
            model="z_image.ZImage/z_image_turbo",
            image=("notes.txt", b"not an image", "text/plain"),
            prompt="anything",
        )


def test_image_edit_lora(client):
    with open(HERE / "car.png", "rb") as car:
        result = client.images.edit(
            model="flux2.variants.Flux2KleinEdit/flux2_klein_4b",
            image=[car],
            prompt="2x2 sprite sheet",
            size=SIZE,
            extra_body={
                "seed": SEED,
                "steps": 4,
                "lora_paths": [
                    "fal/flux-2-klein-4b-spritesheet-lora:flux-spritesheet-lora.safetensors"
                ],
                "lora_scales": [1.1],
            },
        )
    save(result, "car_sprite.png")


def test_image_edit_auto_size_follows_input(client):
    with open(HERE / "car.png", "rb") as car:
        with Image.open(car) as reference:
            expected = ImageEditRequest(model="-", prompt="-").size_kwargs(reference)
        car.seek(0)
        result = client.images.edit(
            model="flux2.variants.Flux2KleinEdit/flux2_klein_4b",
            image=[car],
            prompt="The same car, at night",
            extra_body={
                "seed": SEED,
                "steps": 4,
            },  # no size: the edit default is "auto"
        )
    save(result, "car_night.png", (expected["width"], expected["height"]))


def test_bare_model_name_and_multiple_images(client):
    result = client.images.generate(
        model="flux2_klein_4b",  # expanded server-side to flux2.variants.Flux2Klein/flux2_klein_4b
        prompt=OTTER_PROMPT,
        size=SIZE,
        n=2,
        extra_body={"seed": SEED, "steps": 4},
    )
    assert len(result.data) == 2
    assert result.data[0].b64_json != result.data[1].b64_json  # seed + i per image
