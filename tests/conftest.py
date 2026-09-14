"""Shared fixtures for the API and model tests."""

from __future__ import annotations

import io
from typing import Sequence

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.backends.base import BackendDescriptor, EmotionBackend
from app.config import Settings
from app.emotions import CANONICAL_EMOTIONS
from app.main import create_app


def encode_image(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def make_image(
    size: tuple[int, int] = (96, 96),
    colour: tuple[int, int, int] = (205, 180, 140),
    mouth: tuple[int, int, int] = (70, 45, 35),
) -> Image.Image:
    image = Image.new("RGB", size, colour)
    width, height = size
    draw = ImageDraw.Draw(image)
    draw.rectangle([width // 3, height * 2 // 3, width * 2 // 3, height * 5 // 6], fill=mouth)
    draw.line([(width // 4, height // 3), (width * 3 // 4, height // 3)], fill=(40, 30, 25), width=2)
    return image


class StubBackend(EmotionBackend):
    """Backend whose scores the test picks, so API maths can be checked exactly."""

    def __init__(
        self,
        logits: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        *,
        ready: bool = True,
        model_id: str = "stub/classifier",
        device: str = "cpu",
    ) -> None:
        self._labels = list(labels) if labels is not None else list(CANONICAL_EMOTIONS)
        self._logits = list(logits) if logits is not None else [0.0] * len(self._labels)
        self._ready = ready
        self.calls: list[int] = []
        self._descriptor = BackendDescriptor(
            name="stub",
            model_id=model_id,
            device=device,
            neural=False,
            labels=self._labels,
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict(self, images: Sequence[Image.Image]) -> list[list[float]]:
        self.calls.append(len(images))
        return [list(self._logits) for _ in images]


def reference_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "backend": "reference",
        "warmup_on_start": False,
        "model_id": "reference/pillow-features-v1",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return reference_settings()


@pytest.fixture
def client_factory():
    """Build TestClients that run the app lifespan and shut down afterwards."""

    clients: list[TestClient] = []

    def _make(settings: Settings | None = None, backend: EmotionBackend | None = None) -> TestClient:
        app = create_app(settings or reference_settings(), backend=backend)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make

    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(client_factory) -> TestClient:
    return client_factory()


@pytest.fixture
def stub_client(client_factory):
    def _make(logits: Sequence[float] | None = None, **kwargs: object) -> TestClient:
        return client_factory(backend=StubBackend(logits, **kwargs))

    return _make


@pytest.fixture
def png_bytes() -> bytes:
    return encode_image(make_image())


@pytest.fixture
def jpeg_bytes() -> bytes:
    return encode_image(make_image(), fmt="JPEG")
