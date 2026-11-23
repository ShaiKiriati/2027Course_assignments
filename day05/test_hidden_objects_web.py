import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import hidden_objects_web as web


@pytest.fixture
def training_client(monkeypatch):
    session = web.Session()
    session.image = Image.new("RGB", (8, 8), "white")
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    instance = SimpleNamespace(name="cup", mask=mask)
    session.training_mode = True
    session.training_items = [(instance, "no", ["general", "dishwasher"])]
    session.game_id = 3
    learned_words = []

    def save_word(group, word):
        learned_words.append((group, word))
        return word, True

    session.modules = {"list": SimpleNamespace(save_concept_word=save_word)}
    monkeypatch.setattr(web, "SESSION", session)
    monkeypatch.setitem(web.app.config, "TRAINING_MODE", True)
    monkeypatch.setitem(web.app.config, "TESTING", True)
    with web.app.test_client() as client:
        yield client, session, learned_words


def test_training_page_serves_dedicated_interface(training_client):
    client, _, _ = training_client

    response = client.get("/")

    assert response.status_code == 200
    assert b"CLIP Label Training" in response.data


def test_training_review_returns_labels_and_matching_outline(training_client):
    client, _, _ = training_client

    review = client.get("/api/training").get_json()
    outline_response = client.get("/api/training/outline/3.png")

    assert review["items"][0]["label"] == "cup"
    assert review["items"][0]["status"] == "no"
    assert review["items"][0]["source_groups"] == ["general", "dishwasher"]
    assert review["items"][0]["color"] == "#{:02x}{:02x}{:02x}".format(
        *web.TRAINING_COLORS[0]
    )
    with Image.open(io.BytesIO(outline_response.data)) as outline:
        assert outline.convert("RGBA").getpixel((2, 2)) == (*web.TRAINING_COLORS[0], 255)
    assert client.get("/api/training/outline/2.png").status_code == 404


def test_training_learn_saves_word_to_selected_group_and_updates_review(training_client):
    client, _, learned_words = training_client

    response = client.post(
        "/api/training/learn",
        json={"game_id": 3, "index": 0, "word": "brass key", "group": "dishwasher"},
    )

    assert response.status_code == 200
    assert response.get_json()["added"] is True
    assert response.get_json()["group"] == "dishwasher"
    assert learned_words == [("dishwasher", "brass key")]
    item = client.get("/api/training").get_json()["items"][0]
    assert item["label"] == "brass key"
    assert item["saved"] is True


def test_training_learn_rejects_non_local_client(training_client):
    client, _, learned_words = training_client

    response = client.post(
        "/api/training/learn",
        json={"game_id": 3, "index": 0, "word": "brass key", "group": "dishwasher"},
        environ_overrides={"REMOTE_ADDR": "192.0.2.10"},
    )

    assert response.status_code == 403
    assert learned_words == []


def test_training_learn_rejects_group_not_associated_with_label(training_client):
    client, _, learned_words = training_client

    response = client.post(
        "/api/training/learn",
        json={"game_id": 3, "index": 0, "word": "brass key", "group": "bedroom"},
    )

    assert response.status_code == 400
    assert learned_words == []
