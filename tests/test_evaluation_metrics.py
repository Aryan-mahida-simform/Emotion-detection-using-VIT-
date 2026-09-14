"""Tests for the classification metrics."""

from __future__ import annotations

import pytest

from app.emotions import CANONICAL_EMOTIONS
from app.evaluation.metrics import (
    PredictionRecord,
    confusion_matrix,
    evaluate_predictions,
    per_class_metrics,
    top_k_accuracy,
)

LABELS = ["angry", "happy", "neutral"]


def sample_records() -> list[PredictionRecord]:
    return [
        PredictionRecord("angry", [("angry", 0.9), ("sad", 0.1)], 0.9),
        PredictionRecord("angry", [("sad", 0.6), ("angry", 0.4)], 0.6),
        PredictionRecord("happy", [("happy", 0.8), ("neutral", 0.2)], 0.8),
        PredictionRecord("happy", [("angry", 0.95), ("happy", 0.05)], 0.95),
        PredictionRecord("neutral", [("label_0", 0.7), ("sad", 0.3)], 0.7),
        PredictionRecord("neutral", [], 0.0),
    ]


def test_confusion_matrix_counts_rows_by_expected_label() -> None:
    matrix = confusion_matrix(sample_records(), LABELS)
    assert matrix["angry"] == {"angry": 1, "happy": 0, "neutral": 0, "other": 1}
    assert matrix["happy"]["angry"] == 1
    assert matrix["happy"]["happy"] == 1
    assert matrix["neutral"]["other"] == 1


def test_confusion_matrix_without_unknown_predictions_has_no_other_column() -> None:
    matrix = confusion_matrix(sample_records()[2:4], LABELS)
    assert set(matrix["happy"]) == set(LABELS)


def test_top_k_accuracy_counts_hits_inside_the_ranking() -> None:
    records = sample_records()
    assert top_k_accuracy(records, 1) == pytest.approx(0.4)
    assert top_k_accuracy(records, 2) == pytest.approx(0.8)
    assert top_k_accuracy(records, 7) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        top_k_accuracy(records, 0)


def test_per_class_metrics_match_hand_computed_values() -> None:
    per_class = per_class_metrics(sample_records(), LABELS)
    assert per_class["angry"]["support"] == 2
    assert per_class["angry"]["precision"] == pytest.approx(0.5)
    assert per_class["angry"]["recall"] == pytest.approx(0.5)
    assert per_class["angry"]["f1"] == pytest.approx(0.5)

    assert per_class["happy"]["precision"] == pytest.approx(1.0)
    assert per_class["happy"]["recall"] == pytest.approx(0.5)
    assert per_class["happy"]["f1"] == pytest.approx(2 / 3, abs=1e-6)

    assert per_class["neutral"]["recall"] == pytest.approx(0.0)


def test_full_report_matches_hand_computed_values() -> None:
    report = evaluate_predictions(sample_records(), LABELS, top_k=2)
    assert report["n"] == 5
    assert report["records_total"] == 6
    assert report["accuracy"] == pytest.approx(0.4)
    assert report["top_k"] == {"k": 2, "accuracy": pytest.approx(0.8)}
    assert report["macro"]["precision"] == pytest.approx(0.5)
    assert report["macro"]["recall"] == pytest.approx(1 / 3, abs=1e-6)
    assert report["macro"]["f1"] == pytest.approx(7 / 18, abs=1e-6)
    assert report["weighted"]["precision"] == pytest.approx(0.6)
    assert report["weighted"]["recall"] == pytest.approx(0.4)
    assert report["weighted"]["f1"] == pytest.approx(0.466667, abs=1e-6)
    assert sum(entry["support"] for entry in report["per_class"].values()) == report["n"]
    assert report["mean_confidence"] == pytest.approx(0.79)
    assert report["high_confidence_errors"] == 1
    assert report["labels"] == LABELS


def test_empty_input_returns_a_zeroed_report() -> None:
    report = evaluate_predictions([], CANONICAL_EMOTIONS, top_k=2)
    assert report["n"] == 0
    assert report["accuracy"] == 0.0
    assert report["mean_confidence"] == 0.0
    assert report["high_confidence_errors"] == 0
    assert len(report["per_class"]) == len(CANONICAL_EMOTIONS)
    assert set(report["confusion_matrix"]) == set(CANONICAL_EMOTIONS)


def test_high_confidence_errors_need_a_wrong_answer() -> None:
    records = [PredictionRecord("happy", [("sad", 0.95)], 0.95)]
    report = evaluate_predictions(records, LABELS, high_confidence=0.9)
    assert report["high_confidence_errors"] == 1
    report = evaluate_predictions(records, LABELS, high_confidence=0.99)
    assert report["high_confidence_errors"] == 0
