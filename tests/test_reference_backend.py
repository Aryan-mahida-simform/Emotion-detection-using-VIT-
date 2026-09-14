"""Tests for the deterministic reference backend."""

from __future__ import annotations

import math

import pytest
from PIL import Image, ImageDraw

from app.backends.reference import (
    FEATURE_NAMES,
    ReferenceBackend,
    extract_features,
    score_features,
)
from app.emotions import CANONICAL_EMOTIONS

ZERO_FEATURES = {name: 0.0 for name in FEATURE_NAMES}


def features(**values: float) -> dict[str, float]:
    merged = dict(ZERO_FEATURES)
    merged.update(values)
    return merged


def winner(scores: dict[str, float]) -> str:
    return max(scores, key=lambda label: (scores[label], label))


def test_descriptor_describes_a_non_neural_fallback() -> None:
    descriptor = ReferenceBackend().descriptor
    assert descriptor.name == "reference"
    assert descriptor.neural is False
    assert descriptor.device == "cpu"
    assert descriptor.labels == list(CANONICAL_EMOTIONS)


def test_scores_cover_every_canonical_label() -> None:
    assert set(score_features(ZERO_FEATURES)) == set(CANONICAL_EMOTIONS)


def test_unscored_features_leave_a_neutral_reading() -> None:
    assert winner(score_features(ZERO_FEATURES)) == "neutral"


def test_bright_warm_loose_face_reads_happy() -> None:
    scores = score_features(
        features(brightness=0.4, warmth=0.6, saturation=0.3, mouth_darkness=0.4, edge_density=-0.4, brow_tension=-0.5)
    )
    assert winner(scores) == "happy"


def test_dark_cold_tense_face_reads_angry() -> None:
    scores = score_features(
        features(brightness=-0.5, warmth=-0.3, contrast=0.6, saturation=0.4, edge_density=0.6, brow_tension=0.7, mouth_darkness=0.3)
    )
    assert winner(scores) == "angry"


def test_wide_eyes_and_open_mouth_read_surprise() -> None:
    scores = score_features(
        features(eye_openness=0.9, mouth_darkness=0.8, brow_tension=0.2, brightness=0.1)
    )
    assert winner(scores) == "surprise"


def test_dim_desaturated_face_reads_sad_over_neutral() -> None:
    scores = score_features(
        features(brightness=-0.6, saturation=-0.8, warmth=-0.4, contrast=-0.5, edge_density=-0.2)
    )
    assert winner(scores) == "sad"


def test_unknown_feature_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        score_features({"smile": 1.0})


def test_extract_features_stays_inside_the_centred_range() -> None:
    image = Image.new("RGB", (80, 80), (200, 120, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 55, 60, 70], fill=(20, 10, 10))
    values = extract_features(image).as_dict()
    assert set(values) == set(FEATURE_NAMES)
    for value in values.values():
        assert -1.0 <= value <= 1.0


def test_extract_features_reacts_to_brightness() -> None:
    white = extract_features(Image.new("RGB", (64, 64), (255, 255, 255)))
    black = extract_features(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert white.brightness > 0.9
    assert black.brightness == pytest.approx(-1.0)


def test_extract_features_reads_contrast_from_variation_not_level() -> None:
    flat = Image.new("RGB", (64, 64), (128, 128, 128))
    striped = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(striped)
    for y in range(0, 64, 2):
        draw.line([(0, y), (63, y)], fill=(255, 255, 255), width=1)
    assert extract_features(striped).contrast > extract_features(flat).contrast


def test_extract_features_reacts_to_colour_temperature() -> None:
    warm = extract_features(Image.new("RGB", (32, 32), (240, 160, 60)))
    cool = extract_features(Image.new("RGB", (32, 32), (60, 150, 240)))
    assert warm.warmth > 0.0 > cool.warmth


def test_predict_is_deterministic_and_shaped_like_the_labels() -> None:
    backend = ReferenceBackend()
    image = Image.new("RGB", (48, 48), (210, 170, 130))
    first = backend.predict([image])
    second = backend.predict([image])
    assert first == second
    assert len(first[0]) == len(CANONICAL_EMOTIONS)
    assert all(math.isfinite(value) for value in first[0])


def test_predict_on_no_images_returns_nothing() -> None:
    assert ReferenceBackend().predict([]) == []
