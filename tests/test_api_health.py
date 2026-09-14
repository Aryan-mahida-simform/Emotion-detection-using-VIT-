"""Tests for the health and banner endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.emotions import CANONICAL_EMOTIONS
from app.main import create_app
from tests.conftest import StubBackend, reference_settings


def test_health_reports_the_active_backend(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ready"] is True
    assert payload["backend"] == "reference"
    assert payload["model_id"] == "reference/pillow-features-v1"
    assert payload["uptime_seconds"] >= 0.0
    assert set(payload["upload_modes"]) >= {"base64", "raw"}


def test_readiness_agrees_with_health(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_banner_lists_the_endpoints(client: TestClient) -> None:
    response = client.get("/service")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"]
    assert payload["docs_url"] == "/docs"
    assert payload["endpoints"]["predict_upload"] == "/predict"
    assert payload["endpoints"]["web_app"] == "/"


def test_service_without_a_loaded_backend_reports_degraded() -> None:
    app = create_app(reference_settings())
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ready"] is False
    assert health.json()["status"] == "degraded"

    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "not_ready"


def test_predictions_are_refused_before_the_backend_loads() -> None:
    app = create_app(reference_settings())
    client = TestClient(app)
    response = client.post("/predict/base64", json={"image_base64": "AAAA"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"


def test_health_accepts_an_injected_backend(client_factory) -> None:
    client = client_factory(backend=StubBackend(labels=CANONICAL_EMOTIONS))
    payload = client.get("/health").json()
    assert payload["backend"] == "stub"


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


def test_cors_origins_come_from_settings(client_factory) -> None:
    client = client_factory(
        settings=Settings(backend="reference", warmup_on_start=False, cors_origins=("https://app.test",))
    )
    response = client.get("/health", headers={"Origin": "https://app.test"})
    assert response.headers["access-control-allow-origin"] == "https://app.test"
