"""Probability maths shared by the predictor and the evaluator."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from app.errors import ConfigurationError, ModelOutputError
from app.emotions import expected_affect


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Turn raw scores into probabilities.

    Subtracting the maximum keeps the exponential in range for large logits.
    """

    if temperature <= 0.0:
        raise ConfigurationError("logit_temperature must be greater than zero")
    values = [float(value) for value in logits]
    if not values:
        raise ModelOutputError("backend returned an empty score vector")
    for value in values:
        if not math.isfinite(value):
            raise ModelOutputError(f"backend returned a non-finite score: {value!r}")
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    exponentials = [math.exp(value - peak) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def rank_labels(labels: Sequence[str], probabilities: Sequence[float]) -> list[tuple[str, float]]:
    """Order label/probability pairs by probability, then by label."""

    if len(labels) != len(probabilities):
        raise ModelOutputError(
            f"label count {len(labels)} does not match score count {len(probabilities)}"
        )
    pairs = [(labels[index], float(probabilities[index])) for index in range(len(labels))]
    return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))


def probability_map(labels: Sequence[str], probabilities: Sequence[float]) -> dict[str, float]:
    if len(labels) != len(probabilities):
        raise ModelOutputError(
            f"label count {len(labels)} does not match score count {len(probabilities)}"
        )
    return {labels[index]: float(probabilities[index]) for index in range(len(labels))}


def affect_from_probabilities(probabilities: Mapping[str, float]) -> tuple[float, float]:
    return expected_affect(probabilities)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile, used for latency reporting."""

    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
