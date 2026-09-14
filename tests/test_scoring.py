"""Tests for the probability helpers."""

from __future__ import annotations

import math

import pytest

from app.errors import ConfigurationError, ModelOutputError
from app.scoring import percentile, probability_map, rank_labels, softmax


def test_softmax_is_a_distribution() -> None:
    probabilities = softmax([2.0, 1.0, 0.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_softmax_is_uniform_for_equal_logits() -> None:
    assert softmax([1.5] * 4) == pytest.approx([0.25] * 4)


def test_softmax_survives_large_logits() -> None:
    probabilities = softmax([1000.0, 999.0])
    assert all(math.isfinite(value) for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_temperature_changes_the_shape() -> None:
    sharp = softmax([3.0, 0.0], temperature=0.5)
    flat = softmax([3.0, 0.0], temperature=5.0)
    assert sharp[0] > softmax([3.0, 0.0])[0] > flat[0]


def test_softmax_rejects_bad_input() -> None:
    with pytest.raises(ConfigurationError):
        softmax([1.0, 2.0], temperature=0.0)
    with pytest.raises(ModelOutputError):
        softmax([])
    with pytest.raises(ModelOutputError):
        softmax([float("nan"), 1.0])


def test_rank_labels_orders_by_probability_then_name() -> None:
    ranked = rank_labels(["b", "a", "c"], [0.2, 0.5, 0.3])
    assert ranked == [("a", 0.5), ("c", 0.3), ("b", 0.2)]


def test_rank_labels_breaks_ties_alphabetically() -> None:
    ranked = rank_labels(["sad", "angry"], [0.5, 0.5])
    assert [label for label, _ in ranked] == ["angry", "sad"]


def test_rank_labels_rejects_mismatched_lengths() -> None:
    with pytest.raises(ModelOutputError):
        rank_labels(["a", "b"], [1.0])
    with pytest.raises(ModelOutputError):
        probability_map(["a", "b"], [1.0])


def test_probability_map_keys_follow_the_label_list() -> None:
    mapping = probability_map(["happy", "sad"], [0.75, 0.25])
    assert mapping == {"happy": 0.75, "sad": 0.25}


def test_percentile_interpolates() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([4.0], 0.9) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)
