"""Dependency-free fallback classifier.

This backend computes eight image statistics with Pillow and combines them with
a fixed weight table. It exists so the API, the tests and the evaluation
harness run without torch, transformers, or downloaded weights, and so the
service answers with a labelled, deterministic result instead of failing when
the ViT checkpoint is missing.

It is not a trained emotion model. Treat its output as a placeholder, never as
a prediction about a person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from PIL import Image, ImageFilter, ImageStat

from app.backends.base import BackendDescriptor, EmotionBackend
from app.emotions import CANONICAL_EMOTIONS
from app.imaging import square_resize

FEATURE_SIZE = 64
CONTRAST_SCALE = 64.0
EDGE_SCALE = 32.0
EYE_SCALE = 48.0
LOGIT_SCALE = 2.0
REFERENCE_MODEL_ID = "reference/pillow-features-v1"

FEATURE_NAMES: tuple[str, ...] = (
    "brightness",
    "contrast",
    "saturation",
    "edge_density",
    "warmth",
    "mouth_darkness",
    "eye_openness",
    "brow_tension",
)

# Positive weight means the feature raises that label's score. Features arrive
# centred on [-1, 1], so 0 is the "nothing distinctive here" midpoint.
WEIGHTS: dict[str, dict[str, float]] = {
    "angry": {
        "brightness": -0.7,
        "contrast": 1.1,
        "saturation": 0.6,
        "edge_density": 1.3,
        "warmth": -0.2,
        "mouth_darkness": 0.5,
        "eye_openness": 0.3,
        "brow_tension": 1.3,
    },
    "disgust": {
        "brightness": -0.2,
        "contrast": 0.4,
        "saturation": 0.5,
        "edge_density": 0.7,
        "warmth": -0.7,
        "mouth_darkness": 0.2,
        "eye_openness": 0.0,
        "brow_tension": 0.7,
    },
    "fear": {
        "brightness": -0.4,
        "contrast": 0.9,
        "saturation": 0.2,
        "edge_density": 0.9,
        "warmth": -0.2,
        "mouth_darkness": 0.5,
        "eye_openness": 0.8,
        "brow_tension": 1.0,
    },
    "happy": {
        "brightness": 1.5,
        "contrast": -0.4,
        "saturation": 1.0,
        "edge_density": -0.7,
        "warmth": 2.0,
        "mouth_darkness": 1.1,
        "eye_openness": -0.3,
        "brow_tension": -0.8,
    },
    "neutral": {
        "brightness": 0.2,
        "contrast": -0.5,
        "saturation": -0.8,
        "edge_density": -0.7,
        "warmth": 0.1,
        "mouth_darkness": 0.1,
        "eye_openness": -0.2,
        "brow_tension": -0.8,
    },
    "sad": {
        "brightness": -1.1,
        "contrast": -0.8,
        "saturation": -0.4,
        "edge_density": 0.3,
        "warmth": -0.4,
        "mouth_darkness": -0.2,
        "eye_openness": -0.2,
        "brow_tension": 0.3,
    },
    "surprise": {
        "brightness": 0.7,
        "contrast": 0.5,
        "saturation": 0.4,
        "edge_density": 0.4,
        "warmth": 0.1,
        "mouth_darkness": 1.2,
        "eye_openness": 1.3,
        "brow_tension": 0.5,
    },
}

# Neutral carries the largest bias because a face with no distinctive feature
# should not be pushed towards an expressive class.
BIASES: dict[str, float] = {
    "angry": 0.0,
    "disgust": -0.1,
    "fear": -0.1,
    "happy": 0.0,
    "neutral": 0.45,
    "sad": 0.05,
    "surprise": -0.15,
}


@dataclass(frozen=True)
class ImageFeatures:
    brightness: float
    contrast: float
    saturation: float
    edge_density: float
    warmth: float
    mouth_darkness: float
    eye_openness: float
    brow_tension: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in FEATURE_NAMES}


def _centre(value: float) -> float:
    return max(-1.0, min(1.0, value * 2.0 - 1.0))


def extract_features(image: Image.Image) -> ImageFeatures:
    """Summarise a face crop as eight centred statistics."""

    square = square_resize(image.convert("RGB"), FEATURE_SIZE)
    gray = square.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)

    gray_stats = ImageStat.Stat(gray)
    rgb_stats = ImageStat.Stat(square)
    hsv_stats = ImageStat.Stat(square.convert("HSV"))

    band = FEATURE_SIZE // 3
    brow_band = (band, 0, 2 * band, band)
    mouth_band = (band, 2 * band, 2 * band, FEATURE_SIZE)

    brightness = gray_stats.mean[0] / 255.0
    contrast = min(gray_stats.stddev[0] / CONTRAST_SCALE, 1.0)
    saturation = hsv_stats.mean[1] / 255.0
    edge_density = min(ImageStat.Stat(edges).mean[0] / EDGE_SCALE, 1.0)
    warmth = ((rgb_stats.mean[0] - rgb_stats.mean[2]) / 255.0 + 1.0) / 2.0
    mouth_darkness = 1.0 - ImageStat.Stat(gray.crop(mouth_band)).mean[0] / 255.0
    eye_openness = min(ImageStat.Stat(gray.crop((0, 0, FEATURE_SIZE, band))).stddev[0] / EYE_SCALE, 1.0)
    brow_tension = min(ImageStat.Stat(edges.crop(brow_band)).mean[0] / EDGE_SCALE, 1.0)

    return ImageFeatures(
        brightness=_centre(brightness),
        contrast=_centre(contrast),
        saturation=_centre(saturation),
        edge_density=_centre(edge_density),
        warmth=_centre(warmth),
        mouth_darkness=_centre(mouth_darkness),
        eye_openness=_centre(eye_openness),
        brow_tension=_centre(brow_tension),
    )


def score_features(features: Mapping[str, float]) -> dict[str, float]:
    """Combine centred features into one raw score per canonical emotion."""

    unknown = set(features) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature names: {sorted(unknown)}")
    scores: dict[str, float] = {}
    for label in CANONICAL_EMOTIONS:
        weights = WEIGHTS[label]
        total = BIASES[label]
        for name, value in features.items():
            total += weights[name] * float(value)
        scores[label] = total
    return scores


class ReferenceBackend(EmotionBackend):
    def __init__(self, model_id: str = REFERENCE_MODEL_ID) -> None:
        self._descriptor = BackendDescriptor(
            name="reference",
            model_id=model_id,
            device="cpu",
            neural=False,
            labels=list(CANONICAL_EMOTIONS),
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def score_image(self, image: Image.Image) -> list[float]:
        scores = score_features(extract_features(image).as_dict())
        return [scores[label] * LOGIT_SCALE for label in self.labels]

    def predict(self, images: Sequence[Image.Image]) -> list[list[float]]:
        return [self.score_image(image) for image in images]
