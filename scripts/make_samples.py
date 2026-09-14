"""Write placeholder images so the API and the evaluator run without a dataset.

The output is colour patches, not faces. It exists to exercise the transport,
the endpoints and the metrics code, and any accuracy measured on it says
nothing about emotion recognition.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from app.emotions import CANONICAL_EMOTIONS

DEFAULT_SIZE = 224
DEFAULT_OUTPUT = Path("data/sample")

RECIPES: dict[str, dict[str, object]] = {
    "angry": {"background": (150, 60, 50), "variant": (118, 44, 44), "mouth": (40, 20, 20), "brow": (28, 14, 14), "strokes": 12},
    "disgust": {"background": (122, 132, 72), "variant": (104, 112, 62), "mouth": (52, 58, 32), "brow": (62, 72, 36), "strokes": 8},
    "fear": {"background": (140, 140, 162), "variant": (120, 126, 150), "mouth": (42, 42, 62), "brow": (72, 72, 96), "strokes": 10},
    "happy": {"background": (240, 205, 150), "variant": (234, 196, 142), "mouth": (150, 92, 70), "brow": (212, 172, 120), "strokes": 3},
    "neutral": {"background": (172, 166, 160), "variant": (164, 158, 154), "mouth": (132, 122, 116), "brow": (152, 146, 140), "strokes": 5},
    "sad": {"background": (96, 100, 130), "variant": (86, 92, 126), "mouth": (56, 60, 86), "brow": (70, 76, 106), "strokes": 6},
    "surprise": {"background": (226, 220, 210), "variant": (232, 226, 216), "mouth": (34, 30, 30), "brow": (150, 146, 140), "strokes": 4},
}


def build_image(label: str, *, size: int = DEFAULT_SIZE, variant: int = 0) -> Image.Image:
    recipe = RECIPES[label]
    background = recipe["background"] if variant == 0 else recipe["variant"]
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    band = size // 3
    draw.rectangle([band, 2 * band, size - band, size - band // 2], fill=recipe["mouth"])
    random_source = random.Random(f"{label}-{variant}")
    for _ in range(int(recipe["strokes"])):
        y = random_source.randrange(0, band)
        x_start = random_source.randrange(0, size // 2)
        x_end = x_start + random_source.randrange(band // 2, size // 2)
        draw.line([(x_start, y), (x_end, y)], fill=recipe["brow"], width=2)
    return image


def generate_dataset(
    root: Path,
    *,
    variants: int = 2,
    size: int = DEFAULT_SIZE,
    labels: Sequence[str] = CANONICAL_EMOTIONS,
) -> list[Path]:
    written: list[Path] = []
    for label in labels:
        directory = root / label
        directory.mkdir(parents=True, exist_ok=True)
        for variant in range(variants):
            target = directory / f"{label}-{variant + 1}.png"
            build_image(label, size=size, variant=variant).save(target, format="PNG")
            written.append(target)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_samples",
        description="Write placeholder images into an ImageFolder-style tree.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = generate_dataset(args.output, variants=args.variants, size=args.size)
    print(f"wrote {len(written)} placeholder images to {args.output}")
    print("These are colour patches, not faces. Do not read accuracy from them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
