"""CLI that scores a labelled image directory and reports metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from app.backends.factory import resolve_backend
from app.config import SUPPORTED_BACKENDS, SUPPORTED_DEVICES, Settings, load_settings
from app.emotions import CANONICAL_EMOTIONS
from app.evaluation.runner import evaluate_directory, format_report
from app.logging_config import configure_logging
from app.services.predictor import EmotionPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_model",
        description=(
            "Score every image under DATA_DIR and print accuracy, per-class metrics, "
            "a confusion matrix and latency percentiles. DATA_DIR holds one "
            "subdirectory per emotion."
        ),
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Labelled image directory")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=None)
    parser.add_argument("--model-id", default=None, help="Hugging Face checkpoint id")
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default=None)
    parser.add_argument("--top-k", type=int, default=2, help="k for top-k accuracy")
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap images per label")
    parser.add_argument("--json-out", type=Path, default=None, help="Write the full report here")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Exit with status 1 when accuracy falls below this value",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit downloading checkpoint weights from the Hugging Face hub",
    )
    parser.add_argument("--log-level", default=None)
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {
        "EMOTION_API_BACKEND": args.backend,
        "EMOTION_API_MODEL_ID": args.model_id,
        "EMOTION_API_DEVICE": args.device,
        "EMOTION_API_LOG_LEVEL": args.log_level,
        "EMOTION_API_WARMUP_ON_START": "false",
        "EMOTION_API_ALLOW_MODEL_DOWNLOAD": "true" if args.allow_download else None,
    }
    environ = dict(os.environ)
    environ.update({key: value for key, value in overrides.items() if value is not None})
    return load_settings(environ)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    configure_logging(settings.log_level)

    selection = resolve_backend(settings)
    if selection.is_fallback:
        print(f"notice: reference backend in use ({selection.fallback_reason})", file=sys.stderr)

    predictor = EmotionPredictor(selection.backend, settings, selection)
    try:
        report = evaluate_directory(
            predictor,
            args.data_dir.resolve(),
            top_k=args.top_k,
            max_per_class=args.max_per_class,
            labels=CANONICAL_EMOTIONS,
        )
    finally:
        selection.backend.close()

    print(format_report(report))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written to {args.json_out}")

    accuracy = report["metrics"]["accuracy"]
    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(
            f"FAIL: accuracy {accuracy:.4f} is below the {args.min_accuracy:.4f} threshold",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
