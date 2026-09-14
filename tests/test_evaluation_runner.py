"""Tests for dataset discovery, the evaluation runner and its CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.backends.factory import BackendSelection
from app.backends.reference import ReferenceBackend
from app.emotions import CANONICAL_EMOTIONS
from app.evaluation.runner import discover_dataset, evaluate_directory, format_report
from app.services.predictor import EmotionPredictor
from scripts.evaluate_model import main
from scripts.make_samples import generate_dataset
from tests.conftest import encode_image, make_image, reference_settings


def build_predictor() -> EmotionPredictor:
    settings = reference_settings()
    selection = BackendSelection(backend=ReferenceBackend(), requested="reference")
    return EmotionPredictor(selection.backend, settings, selection)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    generate_dataset(tmp_path, variants=2)
    return tmp_path


def test_generate_samples_writes_one_directory_per_emotion(tmp_path: Path) -> None:
    written = generate_dataset(tmp_path / "sample", variants=2)
    assert len(written) == 2 * len(CANONICAL_EMOTIONS)
    assert all(path.exists() for path in written)
    assert {path.parent.name for path in written} == set(CANONICAL_EMOTIONS)


def test_discover_dataset_reads_label_folders(dataset: Path) -> None:
    items, skipped = discover_dataset(dataset)
    assert len(items) == 2 * len(CANONICAL_EMOTIONS)
    assert skipped == []
    assert {item.label for item in items} == set(CANONICAL_EMOTIONS)


def test_discover_dataset_reports_what_it_skips(dataset: Path) -> None:
    (dataset / "not-an-emotion").mkdir()
    (dataset / "not-an-emotion" / "x.png").write_bytes(b"x")
    (dataset / "happy" / "notes.txt").write_text("ignore me", encoding="utf-8")
    (dataset / ".hidden").mkdir()
    items, skipped = discover_dataset(dataset)
    reasons = {entry["reason"] for entry in skipped}
    assert "unsupported file extension" in reasons
    assert any(reason.startswith("unknown label") for reason in reasons)
    assert all(entry["path"] for entry in skipped)
    assert not any(item.path.name == "notes.txt" for item in items)


def test_discover_dataset_honours_max_per_class(dataset: Path) -> None:
    items, skipped = discover_dataset(dataset, max_per_class=1)
    assert len(items) == len(CANONICAL_EMOTIONS)
    assert sum(1 for entry in skipped if entry["reason"] == "exceeded max_per_class") == len(
        CANONICAL_EMOTIONS
    )


def test_discover_dataset_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_dataset(tmp_path / "absent")


def test_evaluate_directory_reports_metrics_and_latency(dataset: Path) -> None:
    report = evaluate_directory(build_predictor(), dataset, top_k=2)
    assert report["backend"] == "reference"
    assert report["model_id"] == "reference/pillow-features-v1"
    assert report["neural"] is False
    assert report["fallback_reason"] is None
    assert report["images_discovered"] == 2 * len(CANONICAL_EMOTIONS)
    assert report["images_scored"] == report["images_discovered"]
    assert report["failures"] == []
    assert report["metrics"]["n"] == report["images_scored"]
    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0
    assert report["metrics"]["top_k"]["k"] == 2
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["p50"]
    assert report["latency_ms"]["max"] >= report["latency_ms"]["mean"]


def test_evaluate_directory_records_undecodable_files(tmp_path: Path) -> None:
    directory = tmp_path / "happy"
    directory.mkdir()
    (directory / "broken.png").write_bytes(b"not an image")
    (directory / "good.png").write_bytes(encode_image(make_image()))
    report = evaluate_directory(build_predictor(), tmp_path)
    assert report["images_scored"] == 1
    assert len(report["failures"]) == 1
    assert report["failures"][0]["code"] == "invalid_image"


def test_format_report_prints_the_headline_numbers(dataset: Path) -> None:
    text = format_report(evaluate_directory(build_predictor(), dataset))
    assert "accuracy" in text
    assert "macro F1" in text
    assert "neutral" in text


def test_cli_writes_a_json_report(dataset: Path, tmp_path: Path, capsys) -> None:
    target = tmp_path / "report.json"
    exit_code = main(
        ["--data-dir", str(dataset), "--backend", "reference", "--json-out", str(target)]
    )
    assert exit_code == 0
    assert "accuracy" in capsys.readouterr().out
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["metrics"]["n"] == 2 * len(CANONICAL_EMOTIONS)
    assert report["labels"] == list(CANONICAL_EMOTIONS)


def test_cli_fails_when_accuracy_misses_the_threshold(dataset: Path) -> None:
    exit_code = main(["--data-dir", str(dataset), "--backend", "reference", "--min-accuracy", "1.1"])
    assert exit_code == 1


def test_cli_rejects_a_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["--data-dir", str(tmp_path / "absent"), "--backend", "reference"])
