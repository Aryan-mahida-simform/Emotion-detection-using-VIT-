"""Classification metrics for model evaluation.

Implemented with plain Python so the evaluation harness runs in the same
dependency set as the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DIGITS = 6
OTHER_COLUMN = "other"


@dataclass(frozen=True)
class PredictionRecord:
    expected: str
    ranked: list[tuple[str, float]]
    confidence: float = 0.0

    @property
    def predicted(self) -> str:
        return self.ranked[0][0] if self.ranked else ""


def _round(value: float) -> float:
    return round(float(value), DIGITS)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_matrix(
    records: Sequence[PredictionRecord],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    extra = sorted(
        {
            record.predicted
            for record in records
            if record.predicted and record.predicted not in labels
        }
    )
    columns = list(labels) + ([OTHER_COLUMN] if extra else [])
    matrix = {expected: {column: 0 for column in columns} for expected in labels}
    for record in records:
        if not record.predicted or record.expected not in matrix:
            continue
        column = record.predicted if record.predicted in columns else OTHER_COLUMN
        matrix[record.expected][column] += 1
    return matrix


def accuracy(records: Sequence[PredictionRecord]) -> float:
    scored = [record for record in records if record.ranked]
    if not scored:
        return 0.0
    hits = sum(1 for record in scored if record.predicted == record.expected)
    return hits / len(scored)


def top_k_accuracy(records: Sequence[PredictionRecord], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    scored = [record for record in records if record.ranked]
    if not scored:
        return 0.0
    hits = sum(
        1 for record in scored if record.expected in [label for label, _ in record.ranked[:k]]
    )
    return hits / len(scored)


def per_class_metrics(
    records: Sequence[PredictionRecord],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    matrix = confusion_matrix(records, labels)
    report: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = matrix[label].get(label, 0)
        support = sum(matrix[label].values())
        predicted_total = sum(matrix[expected].get(label, 0) for expected in labels)
        precision = _safe_divide(true_positive, predicted_total)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        report[label] = {
            "support": support,
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }
    return report


def _average(
    per_class: dict[str, dict[str, float]],
    metric: str,
    weights: dict[str, float] | None = None,
) -> float:
    if not per_class:
        return 0.0
    if weights is None:
        return _round(sum(entry[metric] for entry in per_class.values()) / len(per_class))
    total = sum(weights.get(label, 0.0) for label in per_class)
    if not total:
        return 0.0
    weighted = sum(entry[metric] * weights.get(label, 0.0) for label, entry in per_class.items())
    return _round(weighted / total)


def evaluate_predictions(
    records: Sequence[PredictionRecord],
    labels: Sequence[str],
    *,
    top_k: int = 2,
    high_confidence: float = 0.9,
) -> dict[str, object]:
    """Summarise a run of predictions against ground truth labels."""

    if not records:
        return {
            "n": 0,
            "records_total": 0,
            "accuracy": 0.0,
            "top_k": {"k": top_k, "accuracy": 0.0},
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "per_class": {label: {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0} for label in labels},
            "confusion_matrix": confusion_matrix([], labels),
            "mean_confidence": 0.0,
            "high_confidence_errors": 0,
            "labels": list(labels),
        }

    scored = [record for record in records if record.ranked]
    per_class = per_class_metrics(records, labels)
    supports = {label: float(entry["support"]) for label, entry in per_class.items()}
    confidences = [record.confidence for record in scored] or [0.0]
    return {
        "n": len(scored),
        "records_total": len(records),
        "accuracy": _round(accuracy(records)),
        "top_k": {"k": top_k, "accuracy": _round(top_k_accuracy(records, top_k))},
        "macro": {
            "precision": _average(per_class, "precision"),
            "recall": _average(per_class, "recall"),
            "f1": _average(per_class, "f1"),
        },
        "weighted": {
            "precision": _average(per_class, "precision", supports),
            "recall": _average(per_class, "recall", supports),
            "f1": _average(per_class, "f1", supports),
        },
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(records, labels),
        "mean_confidence": _round(sum(confidences) / len(confidences)),
        "high_confidence_errors": sum(
            1
            for record in scored
            if record.confidence >= high_confidence and record.predicted != record.expected
        ),
        "labels": list(labels),
    }

