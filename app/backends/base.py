"""Contract every emotion classifier implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from PIL import Image


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    model_id: str
    device: str
    neural: bool
    labels: list[str]


class EmotionBackend(ABC):
    """A classifier that returns one raw score row per image.

    Scores are logits, not probabilities. The predictor applies softmax once,
    so every backend keeps the same contract.
    """

    @property
    @abstractmethod
    def descriptor(self) -> BackendDescriptor: ...

    @property
    def labels(self) -> list[str]:
        return list(self.descriptor.labels)

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def is_ready(self) -> bool:
        """False while a backend still needs to load its weights."""

        return True

    @abstractmethod
    def predict(self, images: Sequence[Image.Image]) -> list[list[float]]:
        """Return a score row per input image, in the declared label order."""

    def close(self) -> None:
        return None
