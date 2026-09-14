"""Tests for the prediction service."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from app.config import Settings
from app.emotions import CANONICAL_EMOTIONS
from app.errors import InvalidImageError, ModelOutputError, PayloadTooLargeError
from app.services.predictor import EmotionPredictor
from tests.conftest import StubBackend, encode_image, make_image, reference_settings


def noisy_image(side: int) -> bytes:
    """Incompressible image bytes, so size limits can be exercised."""

    raw = os.urandom(side * side * 3)
    return encode_image(Image.frombytes("RGB", (side, side), raw))


def build(logits=None, *, settings: Settings | None = None, stubs=None) -> EmotionPredictor:
    backend = stubs or StubBackend(logits)
    return EmotionPredictor(backend, settings or reference_settings())


def test_prediction_reports_a_normalised_distribution() -> None:
    prediction = build([0.0] * 7).predict_bytes(encode_image(make_image()))
    assert prediction.model.backend == "stub"
    assert prediction.label in CANONICAL_EMOTIONS
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert prediction.confidence == pytest.approx(prediction.probabilities[prediction.label])
    assert 0.0 <= prediction.latency_ms


def test_scores_are_ranked_and_limited_by_top_k() -> None:
    prediction = build([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]).predict_bytes(
        encode_image(make_image()), top_k=2
    )
    returned = [score.label for score in prediction.scores]
    ranked = sorted(prediction.probabilities, key=lambda label: -prediction.probabilities[label])
    assert len(returned) == 2
    assert returned == ranked[:2]
    assert returned == [prediction.label, prediction.scores[1].label]
    assert len(prediction.probabilities) == len(CANONICAL_EMOTIONS)


def test_top_k_is_clamped_to_the_label_count() -> None:
    predictor = build()
    assert predictor.resolve_top_k(99) == len(CANONICAL_EMOTIONS)
    assert predictor.resolve_top_k(None) == reference_settings().default_top_k
    assert predictor.resolve_top_k(0) == 1


def test_affect_stays_within_the_circumplex_bounds() -> None:
    prediction = build([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).predict_bytes(encode_image(make_image()))
    assert prediction.label == "angry"
    assert -1.0 <= prediction.affect.valence <= 1.0
    assert 0.0 <= prediction.affect.arousal <= 1.0
    assert prediction.affect.valence < 0.0


def test_confident_logits_produce_a_confident_prediction() -> None:
    prediction = build([12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).predict_bytes(encode_image(make_image()))
    assert prediction.confidence > 0.99


def test_mismatched_score_count_is_a_model_error() -> None:
    with pytest.raises(ModelOutputError):
        build([0.0, 1.0]).predict_bytes(encode_image(make_image()))


def test_invalid_payload_is_rejected_before_the_backend_runs() -> None:
    backend = StubBackend()
    with pytest.raises(InvalidImageError):
        build(stubs=backend).predict_bytes(b"not an image")
    assert backend.calls == []


def test_rejects_payloads_over_the_configured_limit() -> None:
    predictor = build(settings=reference_settings(max_image_bytes=1024))
    with pytest.raises(PayloadTooLargeError):
        predictor.predict_bytes(encode_image(make_image(size=(600, 600))))


def test_predict_images_scores_a_whole_batch() -> None:
    predictions = build([1.0] * 7).predict_images([make_image(), make_image()])
    assert len(predictions) == 2
    assert all(prediction.label in CANONICAL_EMOTIONS for prediction in predictions)


def test_predict_batch_records_successes_and_failures_side_by_side() -> None:
    batch = build([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]).predict_batch(
        [("good.png", encode_image(make_image())), ("bad.png", b"junk")]
    )
    assert batch.count == 2
    assert batch.succeeded == 1
    assert batch.failed == 1
    assert batch.items[0].filename == "good.png"
    assert batch.items[0].prediction is not None
    assert batch.items[0].prediction.label == "happy"
    assert batch.items[1].error is not None
    assert batch.items[1].error.code == "invalid_image"


def test_predict_batch_enforces_the_batch_limit() -> None:
    predictor = build(settings=reference_settings(max_batch_size=1))
    with pytest.raises(PayloadTooLargeError):
        predictor.predict_batch([("a.png", b"x"), ("b.png", b"y")])


def test_warmup_runs_without_a_request() -> None:
    predictor = build()
    latency = predictor.warmup()
    assert latency is not None
    assert latency >= 0.0
    assert predictor.warmup_latency_ms == latency


def test_model_descriptor_matches_the_backend() -> None:
    predictor = build()
    descriptor = predictor.model_descriptor()
    assert descriptor.backend == "stub"
    assert descriptor.labels == list(CANONICAL_EMOTIONS)
    assert predictor.is_ready is True
    assert predictor.requested_backend == "stub"
    assert predictor.fallback_reason is None
