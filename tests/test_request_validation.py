from pathlib import Path

import pytest

from conftest import data_uri

HERE = Path(__file__).parent
ZIMAGE = "z_image.ZImage/z_image_turbo"
CAR = ("car.png", (HERE / "car.png").read_bytes(), "image/png")


def generate(http_client, **body):
    return http_client.post("/v1/images/generations", json={"prompt": "x", **body})


def edit(http_client, files=None, **fields):
    files = [("image[]", CAR)] if files is None else files
    return http_client.post(
        "/v1/images/edits", data={"model": ZIMAGE, "prompt": "x", **fields}, files=files
    )


@pytest.mark.parametrize(
    "body, status",
    [
        ({"model": "nonsense"}, 400),
        ({"model": "flux2.variants.Nope/flux2_klein_4b"}, 400),
        ({"model": ZIMAGE, "image_strength": 0.5}, 400),
        ({"model": ZIMAGE, "size": "big"}, 422),
        ({"model": ZIMAGE, "n": 11}, 422),
        ({"model": ZIMAGE, "lora_paths": ["a", "b"], "lora_scales": [1.0]}, 422),
    ],
    ids=[
        "bad-format",
        "unknown-class",
        "strength-on-generate",
        "bad-size",
        "n-too-big",
        "lora-mismatch",
    ],
)
def test_generation_rejects(http_client, body, status):
    assert generate(http_client, **body).status_code == status


@pytest.mark.parametrize(
    "fields, files, status",
    [
        ({}, [], 400),  # no image
        ({}, [("image[]", CAR), ("mask", CAR)], 400),  # mask unsupported
        ({}, [("image[]", ("notes.txt", b"nope", "text/plain"))], 400),  # not an image
        ({}, [("image[]", CAR), ("image[]", CAR)], 400),  # ZImage takes one image
        ({"image_strength": "1.0"}, None, 400),  # must be < 1
        (
            {"steps": "1", "image_strength": "0.5"},
            None,
            400,
        ),  # int(1 * 0.5) == 0 would ignore the image
        ({"size": "big"}, None, 422),
        ({"lora_paths[]": ["a", "b"], "lora_scales[]": ["1.0"]}, None, 422),
    ],
    ids=[
        "no-image",
        "mask",
        "non-image",
        "too-many",
        "strength-1",
        "strength-skips-all",
        "bad-size",
        "lora-mismatch",
    ],
)
def test_edit_rejects(http_client, fields, files, status):
    assert edit(http_client, files=files, **fields).status_code == status


def test_edit_rejects_oversized_upload(http_client, monkeypatch):
    import malax.server

    monkeypatch.setattr(malax.server, "MAX_UPLOAD_BYTES", 1024)
    assert edit(http_client).status_code == 413


def test_malformed_image_data_uri_is_400(api):
    import openai

    with pytest.raises(openai.BadRequestError):
        api.create(user="what is this?", image_url="data:image/png;base64,!!!")


def test_unreadable_image_data_uri_is_400(api):
    import openai

    with pytest.raises(openai.BadRequestError):
        api.create(user="what is this?", image_url=data_uri("conftest.py", "image/png"))
