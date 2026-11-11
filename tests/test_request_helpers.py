import pytest
from fastapi import HTTPException
from PIL import Image

from malax.mflux_runner import (
    MFLUX_CONFIG_BY_REPO,
    ImageEditRequest,
    expand_image_model,
    models_for_repo,
)
from malax.server import (
    ChatCompletionsRequest,
    ResponsesRequest,
    chat_to_messages,
    decode_data_uri,
    to_chat_messages,
)
from malax.vlm_runner import GenerationStats

FLUX_EDIT = "flux2.variants.Flux2KleinEdit/flux2_klein_4b"


def repo_for(config: str) -> str:
    return next(repo for repo, c in MFLUX_CONFIG_BY_REPO.items() if c == config)


def edit_request(**fields) -> ImageEditRequest:
    return ImageEditRequest(model="-", prompt="-", **fields)


# ---------- size ----------


def test_explicit_size():
    assert edit_request(size="512x768").size_kwargs() == {"width": 512, "height": 768}


def test_auto_size_without_reference_lets_mflux_decide():
    assert edit_request().size_kwargs() == {}


def test_auto_size_caps_at_one_megapixel_and_keeps_aspect():
    # scale = sqrt(1024² / (2048 * 1024)) ≈ 0.707, then rounded down to multiples of 16.
    assert edit_request().size_kwargs(Image.new("RGB", (2048, 1024))) == {
        "width": 1440,
        "height": 720,
    }


def test_auto_size_never_upscales():
    assert edit_request().size_kwargs(Image.new("RGB", (300, 200))) == {
        "width": 288,
        "height": 192,
    }


# ---------- model ids ----------


@pytest.mark.parametrize(
    "name", ["flux2_klein_4b", "flux2-klein-4b", repo_for("flux2_klein_4b")]
)
def test_bare_names_and_repo_ids_expand(name):
    assert expand_image_model(name, "edit") == FLUX_EDIT


@pytest.mark.parametrize("name", [FLUX_EDIT, "someone/custom-lora-merge", "nonsense"])
def test_other_names_pass_through(name):
    assert expand_image_model(name, "generate") == name


def test_models_for_repo():
    tasks = {m.task: m.id for m in models_for_repo(repo_for("flux2_klein_4b"))}
    assert tasks == {
        "generate": "flux2.variants.Flux2Klein/flux2_klein_4b",
        "edit": FLUX_EDIT,
    }
    assert models_for_repo("mlx-community/not-an-image-model") == []


# ---------- data URIs ----------


def test_base64_data_uri_tolerates_line_wrapping():
    assert decode_data_uri("data:text/plain;base64,aGVs\nbG8=") == b"hello"


def test_percent_encoded_data_uri():
    assert decode_data_uri("data:text/plain,hello%20world") == b"hello world"


@pytest.mark.parametrize("uri", ["data:image/png;base64", "data:image/png;base64,!!!"])
def test_malformed_data_uri(uri):
    with pytest.raises(HTTPException) as e:
        decode_data_uri(uri)
    assert e.value.status_code == 400


# ---------- message conversion ----------


def test_chat_messages_flatten_parts_and_map_roles():
    request = ChatCompletionsRequest(
        model="-",
        messages=[
            {"role": "developer", "content": "be brief"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is"},
                    {"type": "image_url", "image_url": {"url": "http://a/1.png"}},
                    {"type": "text", "text": "this?"},
                    {
                        "type": "image_url",
                        "image_url": "http://a/2.png",
                    },  # legacy bare string
                ],
            },
            {"role": "assistant", "content": None},
        ],
    )
    messages, images = chat_to_messages(request)
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "what is\nthis?"},
        {"role": "assistant", "content": ""},
    ]
    assert images == ["http://a/1.png", "http://a/2.png"]


def test_responses_input_puts_instructions_first():
    request = ResponsesRequest(
        model="-",
        instructions="be brief",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": "http://a/1.png"},
                ],
            },
        ],
    )
    messages, images = to_chat_messages(request)
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "what is this?"},
    ]
    assert images == ["http://a/1.png"]


def test_responses_string_input():
    messages, images = to_chat_messages(ResponsesRequest(model="-", input="hi"))
    assert (messages, images) == ([{"role": "user", "content": "hi"}], [])


# ---------- stats ----------


@pytest.mark.parametrize(
    "stats, truncated",
    [
        (
            GenerationStats(tokens=16, finish_reason="stop"),
            False,
        ),  # library verdict wins over the count
        (GenerationStats(tokens=3, finish_reason="length"), True),
        (GenerationStats(tokens=16), True),  # fallback: compare against the limit
        (GenerationStats(tokens=15), False),
    ],
)
def test_truncated(stats, truncated):
    assert stats.truncated(16) is truncated
