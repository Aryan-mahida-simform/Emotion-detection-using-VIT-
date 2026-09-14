"""End-to-end tests for the prediction endpoints."""

from __future__ import annotations

import base64
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.emotions import CANONICAL_EMOTIONS
from app.uploads import multipart_available
from tests.conftest import encode_image, reference_settings

requires_multipart = pytest.mark.skipif(
    not multipart_available(), reason="no multipart parser installed"
)


def noisy_image(side: int = 400) -> bytes:
    raw = os.urandom(side * side * 3)
    return encode_image(Image.frombytes("RGB", (side, side), raw))


def assert_prediction_shape(payload: dict, expected_scores: int = 3) -> None:
    assert payload["label"] in CANONICAL_EMOTIONS
    assert len(payload["scores"]) == expected_scores
    assert len(payload["probabilities"]) == len(CANONICAL_EMOTIONS)
    assert -1.0 <= payload["affect"]["valence"] <= 1.0
    assert 0.0 <= payload["affect"]["arousal"] <= 1.0
    assert payload["latency_ms"] >= 0.0
    assert payload["model"]["backend"] == "reference"


@requires_multipart
def test_predict_upload(client: TestClient, png_bytes: bytes) -> None:
    response = client.post("/predict", files={"file": ("face.png", png_bytes, "image/png")})
    assert response.status_code == 200
    assert_prediction_shape(response.json())


@requires_multipart
def test_predict_upload_respects_top_k(client: TestClient, jpeg_bytes: bytes) -> None:
    response = client.post(
        "/predict",
        params={"top_k": 1},
        files={"file": ("face.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["scores"]) == 1
    assert payload["scores"][0]["label"] == payload["label"]
    assert payload["confidence"] == payload["scores"][0]["probability"]


def test_predict_raw_body(client: TestClient, png_bytes: bytes) -> None:
    response = client.post(
        "/predict/raw", content=png_bytes, headers={"content-type": "image/png"}
    )
    assert response.status_code == 200
    assert_prediction_shape(response.json())


def test_predict_raw_reads_the_declared_content_type(client: TestClient, png_bytes: bytes) -> None:
    rejected = client.post(
        "/predict/raw", content=png_bytes, headers={"content-type": "text/plain"}
    )
    assert rejected.status_code == 415
    assert rejected.json()["error"]["code"] == "unsupported_media_type"

    sniffed = client.post("/predict/raw", content=png_bytes)
    assert sniffed.status_code == 200
    assert_prediction_shape(sniffed.json())


def test_predict_base64_accepts_a_data_url(client: TestClient, png_bytes: bytes) -> None:
    payload = base64.b64encode(png_bytes).decode()
    response = client.post("/predict/base64", json={"image_base64": f"data:image/png;base64,{payload}"})
    assert response.status_code == 200
    assert_prediction_shape(response.json())


def test_predict_base64_rejects_broken_base64(client: TestClient) -> None:
    response = client.post("/predict/base64", json={"image_base64": "%%%not-base64%%%"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


def test_predict_base64_rejects_extra_fields(client: TestClient, png_bytes: bytes) -> None:
    payload = base64.b64encode(png_bytes).decode()
    response = client.post(
        "/predict/base64", json={"image_base64": payload, "colour": "blue"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@requires_multipart
def test_predict_rejects_non_image_content_type(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


@requires_multipart
def test_predict_rejects_undecodable_bytes(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("face.png", b"not really a png", "image/png")})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


@requires_multipart
def test_predict_rejects_oversized_uploads(client_factory) -> None:
    client = client_factory(settings=reference_settings(max_image_bytes=1024))
    response = client.post("/predict", files={"file": ("big.png", noisy_image(), "image/png")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_predict_raw_rejects_oversized_payloads(client_factory) -> None:
    client = client_factory(settings=reference_settings(max_image_bytes=1024))
    response = client.post(
        "/predict/raw", content=noisy_image(), headers={"content-type": "image/png"}
    )
    assert response.status_code == 413


@pytest.mark.parametrize("top_k", [0, 8])
def test_top_k_outside_the_label_range_is_rejected(client: TestClient, png_bytes: bytes, top_k: int) -> None:
    response = client.post(
        "/predict/raw", params={"top_k": top_k}, content=png_bytes, headers={"content-type": "image/png"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@requires_multipart
def test_batch_reports_each_image_separately(client: TestClient, png_bytes: bytes) -> None:
    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("one.png", png_bytes, "image/png")),
            ("files", ("broken.png", b"junk", "image/png")),
            ("files", ("two.png", png_bytes, "image/png")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert payload["succeeded"] == 2
    assert payload["failed"] == 1
    assert payload["items"][0]["prediction"]["label"] in CANONICAL_EMOTIONS
    assert payload["items"][1]["error"]["code"] == "invalid_image"
    assert payload["items"][2]["filename"] == "two.png"


@requires_multipart
def test_batch_enforces_the_configured_limit(client_factory, png_bytes: bytes) -> None:
    client = client_factory(settings=reference_settings(max_batch_size=1))
    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("one.png", png_bytes, "image/png")),
            ("files", ("two.png", png_bytes, "image/png")),
        ],
    )
    assert response.status_code == 413


def test_stub_backend_drives_exactly_computed_probabilities(stub_client, png_bytes: bytes) -> None:
    client = stub_client([0.0] * 7)
    payload = client.post(
        "/predict/raw", content=png_bytes, headers={"content-type": "image/png"}
    ).json()
    assert payload["model"]["backend"] == "stub"
    assert payload["label"] == "angry"
    assert payload["confidence"] == pytest.approx(1 / 7, abs=1e-6)
    assert all(value == pytest.approx(1 / 7, abs=1e-6) for value in payload["probabilities"].values())


def test_stub_backend_confidence_follows_the_logits(stub_client, png_bytes: bytes) -> None:
    client = stub_client([12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    payload = client.post(
        "/predict/raw", content=png_bytes, headers={"content-type": "image/png"}
    ).json()
    assert payload["label"] == "angry"
    assert payload["confidence"] > 0.99
    assert payload["affect"]["valence"] < 0


def test_unready_backend_answers_503(stub_client, png_bytes: bytes) -> None:
    client = stub_client([0.0] * 7, ready=False)
    response = client.post("/predict/raw", content=png_bytes, headers={"content-type": "image/png"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"
