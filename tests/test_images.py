import base64
from pathlib import Path

SEED = 1729
SIZE = "512x512"

HERE = Path(__file__).parent
OUTPUT = HERE / "output"

OTTER_PROMPT = """
A children's book drawing of a veterinarian using a stethoscope to
listen to the heartbeat of a baby otter.
"""


def save(result, name: str) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / name
    path.write_bytes(base64.b64decode(result.data[0].b64_json))
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

    with open(HERE / "apple.jpg", "rb") as apple, open(HERE / "vases.jpg", "rb") as vases:
        result = client.images.edit(
            model="flux2.variants.Flux2KleinEdit/flux2_klein_4b",
            image=[apple, vases],
            prompt=prompt,
            size=SIZE,
            extra_body={"seed": SEED, "steps": 4},
        )
    save(result, "painting.png")


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
                "lora_paths": ["fal/flux-2-klein-4b-spritesheet-lora:flux-spritesheet-lora.safetensors"],
                "lora_scales": [1.1],
            },
        )
    save(result, "car_sprite.png")
