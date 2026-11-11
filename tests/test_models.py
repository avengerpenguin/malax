import pytest

from conftest import MODEL

ZIMAGE = "z_image.ZImage/z_image_turbo"


def test_list_models(client):
    models = client.models.list()
    model_names = [model.id for model in models.data]
    assert MODEL in model_names
    assert ZIMAGE in model_names


def test_image_models_advertise_their_endpoint(client):
    entries = {m.id: m for m in client.models.list().data}
    assert entries[ZIMAGE].endpoint == "/v1/images/generations"
    assert getattr(entries[MODEL], "endpoint", None) is None


def test_retrieve_model(http_client):
    # Plain HTTP: the slash in repo ids is the point, and SDKs may percent-encode it.
    assert http_client.get(f"/v1/models/{MODEL}").json()["id"] == MODEL
    assert http_client.get("/v1/models/nobody/nothing").status_code == 404


def test_health_and_ui(http_client):
    assert http_client.get("/health").json()["status"] == "ok"
    ui = http_client.get("/")
    assert ui.status_code == 200 and "<main" in ui.text


@pytest.mark.slow
def test_switches_between_chat_and_image_models(client):
    def active():
        return [m.id for m in client.models.list().data if getattr(m, "active", False)]

    def ask():
        return client.chat.completions.create(
            model=MODEL, max_tokens=8, messages=[{"role": "user", "content": "Say hi."}]
        )

    ask()
    assert active() == [MODEL]

    client.images.generate(
        model=ZIMAGE, prompt="a red square", size="256x256", extra_body={"steps": 2}
    )
    # Image models are keyed by JSON; active_model_id must unwrap it.
    assert active() == [ZIMAGE]

    assert ask().choices[0].message.content  # and back again, after unload
