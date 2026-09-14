"""The contract every backend must satisfy, and how the factory picks one."""

from __future__ import annotations

import math
from typing import Sequence

import pytest
from PIL import Image

from app.backends.base import BackendDescriptor, EmotionBackend
from app.backends.factory import resolve_backend
from app.backends.reference import ReferenceBackend
from app.backends.vit import ViTBackend
from app.config import Settings
from app.errors import ApiError, BackendUnavailableError
from app.main import create_app
from app.services.predictor import EmotionPredictor
from tests.conftest import StubBackend, make_image, reference_settings


def backend_cases() -> Sequence[BackendDescriptor]:
    return [
        pytest.param(ReferenceBackend, id="reference"),
        pytest.param(StubBackend, id="stub"),
    ]


@pytest.mark.parametrize("factory", backend_cases())
def test_backend_contract(factory) -> None:
    backend: EmotionBackend = factory()
    descriptor = backend.descriptor
    assert descriptor.name
    assert descriptor.model_id
    assert descriptor.device
    assert isinstance(descriptor.neural, bool)

    labels = backend.labels
    assert labels
    assert len(set(labels)) == len(labels)
    assert backend.is_ready is True
    assert backend.predict([]) == []

    rows = backend.predict([make_image(), make_image(size=(64, 64))])
    assert len(rows) == 2
    for row in rows:
        assert len(row) == len(labels)
        assert all(math.isfinite(value) for value in row)


def test_reference_backend_is_selected_by_name() -> None:
    selection = resolve_backend(reference_settings())
    assert isinstance(selection.backend, ReferenceBackend)
    assert selection.requested == "reference"
    assert selection.fallback_reason is None
    assert selection.is_fallback is False


def test_auto_falls_back_with_a_reason_when_the_checkpoint_is_unreachable() -> None:
    settings = Settings(
        backend="auto",
        model_id="simform-emotion-detection/not-a-real-checkpoint",
        allow_model_download=False,
        warmup_on_start=False,
    )
    selection = resolve_backend(settings)
    assert isinstance(selection.backend, ReferenceBackend)
    assert selection.fallback_reason


def test_explicit_vit_choice_fails_instead_of_degrading() -> None:
    settings = Settings(
        backend="vit",
        model_id="simform-emotion-detection/not-a-real-checkpoint",
        allow_model_download=False,
        warmup_on_start=False,
    )
    with pytest.raises(ApiError):
        resolve_backend(settings)


def test_unavailable_backend_reports_itself_through_the_api() -> None:
    selector = ViTBackend("simform-emotion-detection/not-a-real-checkpoint", local_files_only=True)
    with pytest.raises(BackendUnavailableError):
        selector.load()
    assert selector.is_ready is False


def test_predictor_exposes_the_fallback_reason() -> None:
    settings = Settings(
        backend="auto",
        model_id="simform-emotion-detection/not-a-real-checkpoint",
        allow_model_download=False,
        warmup_on_start=False,
    )
    selection = resolve_backend(settings)
    predictor = EmotionPredictor(selection.backend, settings, selection)
    assert predictor.fallback_reason is not None
    assert predictor.requested_backend == "auto"
    assert predictor.is_ready is True


def test_app_falls_back_instead_of_refusing_to_start() -> None:
    settings = Settings(
        backend="auto",
        model_id="simform-emotion-detection/not-a-real-checkpoint",
        allow_model_download=False,
        warmup_on_start=False,
    )
    application = create_app(settings)
    assert application.state.settings is settings
    assert application.state.upload_modes


def test_reference_backend_ignores_a_checkpoint_id() -> None:
    backend = ReferenceBackend(model_id="custom/id")
    assert backend.descriptor.model_id == "custom/id"
    assert backend.predict([Image.new("RGB", (16, 16), (10, 10, 10))])[0]
