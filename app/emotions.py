"""Emotion label vocabulary shared by every backend and endpoint.

The seven canonical labels are the FER2013 classes. Public ViT checkpoints for
facial emotion are fine-tuned on that set, so the API normalises whatever the
checkpoint reports to these names.
"""

from __future__ import annotations

from typing import Mapping, Sequence

CANONICAL_EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

LABEL_ALIASES: dict[str, str] = {
    "anger": "angry",
    "angry": "angry",
    "rage": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "contempt": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "scared": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "joyful": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "sorrow": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}

# Circumplex coordinates from Russell (1980), "A circumplex model of affect".
# Valence spans [-1, 1], arousal spans [0, 1]. The API uses these fixed points
# to turn a discrete distribution into the two continuous scores it reports.
AFFECT_PROFILE: dict[str, tuple[float, float]] = {
    "angry": (-0.62, 0.80),
    "disgust": (-0.70, 0.50),
    "fear": (-0.70, 0.85),
    "happy": (0.80, 0.60),
    "neutral": (0.00, 0.20),
    "sad": (-0.60, 0.25),
    "surprise": (0.10, 0.90),
}

_CANONICAL_INDEX: dict[str, int] = {label: i for i, label in enumerate(CANONICAL_EMOTIONS)}


def normalize_label(raw: str) -> str:
    """Map a backend or dataset label onto a stable lowercase slug."""

    if not isinstance(raw, str):
        raise TypeError(f"label must be a string, got {type(raw).__name__}")
    collapsed = " ".join(raw.replace("_", " ").replace("-", " ").split()).lower()
    if not collapsed:
        raise ValueError("label must not be blank")
    slug = collapsed.replace(" ", "_")
    return LABEL_ALIASES.get(slug, slug)


def is_canonical(label: str) -> bool:
    return label in _CANONICAL_INDEX


def fallback_labels(num_labels: int) -> list[str]:
    if num_labels <= 0:
        raise ValueError("num_labels must be positive")
    if num_labels == len(CANONICAL_EMOTIONS):
        return list(CANONICAL_EMOTIONS)
    return [f"label_{index}" for index in range(num_labels)]


def _dedupe(labels: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label if count == 0 else f"{label}_{count + 1}")
    return unique


def labels_from_id2label(id2label: Mapping[int | str, str] | None, num_labels: int) -> list[str]:
    """Read a checkpoint's label map, ordered by output index.

    Falls back to the canonical set when the map is missing, short, or holds
    duplicate names after normalisation.
    """

    if not id2label:
        return fallback_labels(num_labels)
    ordered: list[str] = []
    for index in range(num_labels):
        raw = id2label.get(index, id2label.get(str(index)))
        if raw is None:
            return fallback_labels(num_labels)
        ordered.append(normalize_label(str(raw)))
    if len(set(ordered)) != len(ordered):
        return fallback_labels(num_labels)
    return ordered


def expected_affect(probabilities: Mapping[str, float]) -> tuple[float, float]:
    """Return the distribution's mean valence and arousal.

    Labels without a circumplex entry are ignored, and the mean is taken over
    the probability mass that remains.
    """

    total = 0.0
    valence = 0.0
    arousal = 0.0
    for label, probability in probabilities.items():
        profile = AFFECT_PROFILE.get(label)
        if profile is None:
            continue
        weight = float(probability)
        valence += profile[0] * weight
        arousal += profile[1] * weight
        total += weight
    if total <= 0.0:
        return 0.0, 0.0
    return valence / total, arousal / total
