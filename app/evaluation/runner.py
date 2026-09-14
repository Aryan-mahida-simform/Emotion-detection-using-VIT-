"""Run the active backend over a labelled directory and report metrics."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.emotions import CANONICAL_EMOTIONS, normalize_label
from app.errors import ApiError
from app.evaluation.metrics import PredictionRecord, evaluate_predictions
from app.scoring import percentile
from app.services.predictor import EmotionPredictor

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)


@dataclass(frozen=True)
class DatasetItem:
    path: Path
    label: str


def discover_dataset(
    root: Path,
    *,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
    max_per_class: int | None = None,
) -> tuple[list[DatasetItem], list[dict[str, str]]]:
    """Read an ImageFolder-style tree: one subdirectory per emotion."""

    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    allowed = set(labels)
    items: list[DatasetItem] = []
    skipped: list[dict[str, str]] = []

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if directory.name.startswith("."):
            continue
        try:
            label = normalize_label(directory.name)
        except ValueError:
            skipped.append({"path": str(directory), "reason": "blank directory name"})
            continue
        if label not in allowed:
            skipped.append({"path": str(directory), "reason": f"unknown label {label!r}"})
            continue
        taken = 0
        for file in sorted(path for path in directory.iterdir() if path.is_file()):
            if file.suffix.lower() not in IMAGE_SUFFIXES:
                skipped.append({"path": str(file), "reason": "unsupported file extension"})
                continue
            if max_per_class is not None and taken >= max_per_class:
                skipped.append({"path": str(file), "reason": "exceeded max_per_class"})
                continue
            items.append(DatasetItem(path=file, label=label))
            taken += 1
    return items, skipped


def evaluate_directory(
    predictor: EmotionPredictor,
    root: Path,
    *,
    top_k: int = 2,
    max_per_class: int | None = None,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
) -> dict[str, Any]:
    """Score every discovered image and return a JSON-serialisable report."""

    items, skipped = discover_dataset(root, labels=labels, max_per_class=max_per_class)
    descriptor = predictor.backend.descriptor
    records: list[PredictionRecord] = []
    failures: list[dict[str, str]] = []
    latencies: list[float] = []

    for item in items:
        try:
            prediction = predictor.predict_bytes(item.path.read_bytes(), media_type=None)
        except ApiError as exc:
            failures.append({"path": str(item.path), "code": exc.code, "message": exc.message})
            continue
        except OSError as exc:
            failures.append({"path": str(item.path), "code": "unreadable_file", "message": str(exc)})
            continue
        ranked = sorted(prediction.probabilities.items(), key=lambda pair: (-pair[1], pair[0]))
        records.append(
            PredictionRecord(
                expected=item.label,
                ranked=list(ranked),
                confidence=prediction.confidence,
            )
        )
        latencies.append(prediction.latency_ms)

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "dataset_dir": str(root),
        "labels": list(labels),
        "backend": descriptor.name,
        "model_id": descriptor.model_id,
        "device": descriptor.device,
        "neural": descriptor.neural,
        "fallback_reason": predictor.fallback_reason,
        "images_discovered": len(items),
        "images_scored": len(records),
        "images_skipped": len(skipped),
        "skipped": skipped,
        "failures": failures,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.5), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "metrics": evaluate_predictions(records, labels, top_k=top_k),
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"dataset        {report['dataset_dir']}",
        f"backend        {report['backend']} ({report['model_id']})",
        f"fallback       {report['fallback_reason'] or 'none'}",
        f"images         {report['images_scored']} scored, {report['images_skipped']} skipped, "
        f"{len(report['failures'])} failed",
        "",
        f"accuracy       {metrics['accuracy']:.4f}",
        f"top-{metrics['top_k']['k']} accuracy  {metrics['top_k']['accuracy']:.4f}",
        f"macro F1       {metrics['macro']['f1']:.4f}",
        f"weighted F1    {metrics['weighted']['f1']:.4f}",
        f"mean conf.     {metrics['mean_confidence']:.4f}",
        f"confident errs {metrics['high_confidence_errors']}",
        f"latency p50    {report['latency_ms']['p50']:.2f} ms",
        f"latency p95    {report['latency_ms']['p95']:.2f} ms",
        "",
        f"{'label':<10}{'support':>8}{'precision':>11}{'recall':>9}{'f1':>9}",
    ]
    for label, entry in metrics["per_class"].items():
        lines.append(
            f"{label:<10}{entry['support']:>8}{entry['precision']:>11.4f}"
            f"{entry['recall']:>9.4f}{entry['f1']:>9.4f}"
        )
    return "\n".join(lines)
