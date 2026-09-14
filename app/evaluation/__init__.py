"""Model evaluation tooling."""

from app.evaluation.metrics import (
    PredictionRecord,
    confusion_matrix,
    evaluate_predictions,
    per_class_metrics,
    top_k_accuracy,
)
from app.evaluation.runner import discover_dataset, evaluate_directory, format_report

__all__ = [
    "PredictionRecord",
    "confusion_matrix",
    "discover_dataset",
    "evaluate_directory",
    "evaluate_predictions",
    "format_report",
    "per_class_metrics",
    "top_k_accuracy",
]
