"""Tests for the model metadata endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.emotions import CANONICAL_EMOTIONS
from tests.conftest import StubBackend


def test_model_info_describes_the_reference_backend(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "reference"
    assert payload["requested_backend"] == "reference"
    assert payload["neural"] is False
    assert payload["fallback_reason"] is None
    assert payload["labels"] == list(CANONICAL_EMOTIONS)
    assert payload["canonical_labels"] == list(CANONICAL_EMOTIONS)
    assert payload["image_size"] == 224
    assert payload["logit_temperature"] == 1.0
    assert payload["default_top_k"] == 3
    assert set(payload["upload_modes"]) >= {"base64", "raw"}
    assert "image/jpeg" in payload["allowed_media_types"]


def test_model_info_reports_the_injected_backend(client_factory) -> None:
    client = client_factory(
        backend=StubBackend(logits=[0.0] * 7, model_id="stub/seven-class", device="cuda")
    )
    payload = client.get("/model/info").json()
    assert payload["backend"] == "stub"
    assert payload["model_id"] == "stub/seven-class"
    assert payload["device"] == "cuda"
    assert payload["labels"] == list(CANONICAL_EMOTIONS)


def test_model_labels_exposes_aliases_and_affect_anchors(client: TestClient) -> None:
    payload = client.get("/model/labels").json()
    assert payload["labels"] == list(CANONICAL_EMOTIONS)
    assert payload["label_aliases"]["happiness"] == "happy"
    assert set(payload["affect_profile"]) == set(CANONICAL_EMOTIONS)
    valence, arousal = payload["affect_profile"]["happy"]
    assert valence > 0.0
    assert 0.0 <= arousal <= 1.0
