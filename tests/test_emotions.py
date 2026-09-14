"""Tests for the emotion label vocabulary."""

from __future__ import annotations

import pytest

from app.emotions import (
    AFFECT_PROFILE,
    CANONICAL_EMOTIONS,
    expected_affect,
    fallback_labels,
    is_canonical,
    labels_from_id2label,
    normalize_label,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("happy", "happy"),
        ("HAPPY", "happy"),
        ("  Neutral  ", "neutral"),
        ("happiness", "happy"),
        ("fearful", "fear"),
        ("surprised", "surprise"),
        ("SADNESS", "sad"),
        ("angry-face", "angry_face"),
        ("angry_face", "angry_face"),
        ("contempt", "disgust"),
    ],
)
def test_normalize_label_maps_variants(raw: str, expected: str) -> None:
    assert normalize_label(raw) == expected


def test_normalize_label_rejects_blank_and_non_string() -> None:
    with pytest.raises(ValueError):
        normalize_label("   ")
    with pytest.raises(TypeError):
        normalize_label(7)  # type: ignore[arg-type]


def test_is_canonical_only_accepts_the_fer_set() -> None:
    assert all(is_canonical(label) for label in CANONICAL_EMOTIONS)
    assert not is_canonical("contempt")
    assert not is_canonical("label_3")


def test_fallback_labels_uses_the_fer_set_at_width_seven() -> None:
    assert fallback_labels(7) == list(CANONICAL_EMOTIONS)
    assert fallback_labels(3) == ["label_0", "label_1", "label_2"]
    with pytest.raises(ValueError):
        fallback_labels(0)


def test_labels_from_id2label_reads_a_checkpoint_map() -> None:
    id2label = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happiness",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }
    assert labels_from_id2label(id2label, 7) == list(CANONICAL_EMOTIONS)


def test_labels_from_id2label_accepts_string_keys() -> None:
    id2label = {str(index): label for index, label in enumerate(CANONICAL_EMOTIONS)}
    assert labels_from_id2label(id2label, 7) == list(CANONICAL_EMOTIONS)


def test_labels_from_id2label_falls_back_when_incomplete() -> None:
    assert labels_from_id2label(None, 7) == list(CANONICAL_EMOTIONS)
    assert labels_from_id2label({0: "happy"}, 4) == ["label_0", "label_1", "label_2", "label_3"]


def test_labels_from_id2label_rejects_duplicates() -> None:
    id2label = {index: "happy" for index in range(7)}
    assert labels_from_id2label(id2label, 7) == list(CANONICAL_EMOTIONS)


def test_affect_profile_covers_every_canonical_emotion() -> None:
    assert set(AFFECT_PROFILE) == set(CANONICAL_EMOTIONS)
    for valence, arousal in AFFECT_PROFILE.values():
        assert -1.0 <= valence <= 1.0
        assert 0.0 <= arousal <= 1.0


def test_expected_affect_returns_the_anchor_for_a_certain_prediction() -> None:
    valence, arousal = expected_affect({"happy": 1.0})
    assert (valence, arousal) == pytest.approx(AFFECT_PROFILE["happy"])


def test_expected_affect_averages_and_ignores_unknown_labels() -> None:
    valence, arousal = expected_affect({"happy": 0.5, "sad": 0.5, "label_0": 0.9})
    assert valence == pytest.approx((AFFECT_PROFILE["happy"][0] + AFFECT_PROFILE["sad"][0]) / 2)
    assert arousal == pytest.approx((AFFECT_PROFILE["happy"][1] + AFFECT_PROFILE["sad"][1]) / 2)


def test_expected_affect_of_unknown_labels_only_is_neutral_zero() -> None:
    assert expected_affect({"label_0": 1.0}) == (0.0, 0.0)
