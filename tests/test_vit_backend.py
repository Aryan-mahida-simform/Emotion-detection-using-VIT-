"""Tests for the Vision Transformer adapter.

The checkpoint tests are marked ``vit`` and need torch, transformers and
weights on disk. Run them with ``pytest -m vit`` after installing
requirements-vit.txt and setting EMOTION_API_TEST_ALLOW_DOWNLOAD=1.
"""

from __future__ import annotations

import math
import os

import pytest
from PIL import Image

from app.backends.vit import ViTBackend
from app.errors import BackendUnavailableError, ConfigurationError
from app.scoring import softmax
from tests.conftest import make_image

MISSING_CHECKPOINT = "simform-emotion-detection/not-a-real-checkpoint"


def test_dependency_probe_always_returns_a_reason() -> None:
    available, reason = ViTBackend.dependencies_available()
    assert isinstance(available, bool)
    assert isinstance(reason, str)
    if not available:
        assert reason


def test_backend_starts_unloaded_and_refuses_to_predict() -> None:
    backend = ViTBackend("org/checkpoint")
    assert backend.is_ready is False
    assert backend.descriptor.labels == []
    with pytest.raises(BackendUnavailableError):
        backend.predict([make_image(size=(32, 32))])


def test_load_reports_a_missing_checkpoint_as_unavailable() -> None:
    backend = ViTBackend(MISSING_CHECKPOINT, local_files_only=True)
    with pytest.raises(BackendUnavailableError) as error:
        backend.load()
    assert backend.is_ready is False
    assert error.value.details["model_id"] == MISSING_CHECKPOINT


def test_explicit_cuda_without_a_gpu_is_a_configuration_error() -> None:
    available, _ = ViTBackend.dependencies_available()
    if not available:
        pytest.skip("torch is not installed")
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a CUDA device")
    backend = ViTBackend(MISSING_CHECKPOINT, device="cuda", local_files_only=True)
    with pytest.raises(ConfigurationError):
        backend.load()


def _test_model_id() -> str:
    return os.environ.get("EMOTION_API_TEST_MODEL_ID", "")


@pytest.mark.vit
def test_real_checkpoint_predicts_over_the_label_set() -> None:
    model_id = _test_model_id()
    if not model_id:
        pytest.skip("set EMOTION_API_TEST_MODEL_ID to a cached checkpoint id")
    allow_download = os.environ.get("EMOTION_API_TEST_ALLOW_DOWNLOAD") == "1"

    backend = ViTBackend(model_id, device="cpu", local_files_only=not allow_download)
    backend.load()

    labels = backend.labels
    assert labels
    assert backend.is_ready is True
    assert backend.descriptor.neural is True

    rows = backend.predict([make_image(size=(224, 224))])
    assert len(rows) == 1
    assert len(rows[0]) == len(labels)
    assert all(math.isfinite(value) for value in rows[0])
    probabilities = softmax(rows[0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert max(probabilities) > 0.0


@pytest.mark.vit
def test_real_checkpoint_scores_a_batch() -> None:
    model_id = _test_model_id()
    if not model_id:
        pytest.skip("set EMOTION_API_TEST_MODEL_ID to a cached checkpoint id")

    backend = ViTBackend(model_id, device="cpu", local_files_only=True)
    backend.load()
    images = [make_image(size=(224, 224)), Image.new("RGB", (224, 224), (30, 30, 30))]
    rows = backend.predict(images)
    assert len(rows) == len(images)
    assert all(len(row) == len(backend.labels) for row in rows)
