"""Prediction service: decode, classify, and shape the response."""

from __future__ import annotations

import logging
import time
from typing import Sequence

from PIL import Image

from app.backends.base import EmotionBackend
from app.backends.factory import BackendSelection
from app.config import Settings
from app.errors import ApiError, ModelOutputError, NotReadyError, PayloadTooLargeError
from app.imaging import check_media_type, check_size, decode_image
from app.schemas import Affect, BatchItem, BatchPrediction, EmotionScore, ModelDescriptor, Prediction
from app.scoring import affect_from_probabilities, probability_map, rank_labels, softmax

logger = logging.getLogger(__name__)

PROBABILITY_DIGITS = 6
WARMUP_IMAGE_COLOUR = (128, 128, 128)


class EmotionPredictor:
    """Runs the backend and turns raw scores into API payloads."""

    def __init__(
        self,
        backend: EmotionBackend,
        settings: Settings,
        selection: BackendSelection | None = None,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._selection = selection
        self._warmup_latency_ms: float | None = None

    @property
    def backend(self) -> EmotionBackend:
        return self._backend

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def selection(self) -> BackendSelection | None:
        return self._selection

    @property
    def labels(self) -> list[str]:
        return self._backend.labels

    @property
    def is_ready(self) -> bool:
        return self._backend.is_ready

    @property
    def fallback_reason(self) -> str | None:
        return self._selection.fallback_reason if self._selection else None

    @property
    def requested_backend(self) -> str:
        return self._selection.requested if self._selection else self._backend.name

    @property
    def warmup_latency_ms(self) -> float | None:
        return self._warmup_latency_ms

    def model_descriptor(self) -> ModelDescriptor:
        descriptor = self._backend.descriptor
        return ModelDescriptor(
            backend=descriptor.name,
            model_id=descriptor.model_id,
            device=descriptor.device,
            neural=descriptor.neural,
            labels=list(descriptor.labels),
        )

    def warmup(self) -> float | None:
        """Run one image through the backend so the first request is not the slow one."""

        size = self._settings.image_size
        probe = Image.new("RGB", (size, size), WARMUP_IMAGE_COLOUR)
        started = time.perf_counter()
        try:
            self._backend.predict([probe])
        except Exception:
            logger.exception("warmup failed for backend %s", self._backend.name)
            return None
        self._warmup_latency_ms = (time.perf_counter() - started) * 1000.0
        return self._warmup_latency_ms

    def require_ready(self) -> None:
        """Refuse to answer while the backend still has no weights."""

        if not self._backend.is_ready:
            raise NotReadyError(
                f"backend {self._backend.name} has not finished loading",
                details={"model_id": self._backend.descriptor.model_id},
            )

    def resolve_top_k(self, top_k: int | None) -> int:
        requested = self._settings.default_top_k if top_k is None else top_k
        return max(1, min(int(requested), len(self.labels)))

    def build_prediction(
        self,
        logits: Sequence[float],
        *,
        top_k: int | None = None,
        latency_ms: float = 0.0,
    ) -> Prediction:
        labels = self.labels
        if len(logits) != len(labels):
            raise ModelOutputError(
                f"backend returned {len(logits)} scores for {len(labels)} labels",
                details={"model_id": self._backend.descriptor.model_id},
            )
        probabilities = softmax(logits, self._settings.logit_temperature)
        ranked = rank_labels(labels, probabilities)
        per_label = probability_map(labels, probabilities)
        valence, arousal = affect_from_probabilities(per_label)
        limit = self.resolve_top_k(top_k)

        return Prediction(
            label=ranked[0][0],
            confidence=round(ranked[0][1], PROBABILITY_DIGITS),
            scores=[
                EmotionScore(label=label, probability=round(probability, PROBABILITY_DIGITS))
                for label, probability in ranked[:limit]
            ],
            probabilities={
                label: round(probability, PROBABILITY_DIGITS) for label, probability in ranked
            },
            affect=Affect(
                valence=round(valence, PROBABILITY_DIGITS),
                arousal=round(arousal, PROBABILITY_DIGITS),
            ),
            model=self.model_descriptor(),
            latency_ms=round(latency_ms, 3),
        )

    def predict_bytes(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        top_k: int | None = None,
    ) -> Prediction:
        self.require_ready()
        check_media_type(media_type, self._settings.accepts_media_type)
        check_size(data, self._settings.max_image_bytes)
        started = time.perf_counter()
        image = decode_image(data, max_side=self._settings.max_image_side)
        logits = self._backend.predict([image])[0]
        elapsed = (time.perf_counter() - started) * 1000.0
        return self.build_prediction(logits, top_k=top_k, latency_ms=elapsed)

    def predict_images(
        self,
        images: Sequence[Image.Image],
        *,
        top_k: int | None = None,
    ) -> list[Prediction]:
        if not images:
            return []
        self.require_ready()
        started = time.perf_counter()
        rows = self._backend.predict(images)
        elapsed = (time.perf_counter() - started) * 1000.0
        per_image = elapsed / len(rows) if rows else 0.0
        return [self.build_prediction(row, top_k=top_k, latency_ms=per_image) for row in rows]

    def predict_batch(
        self,
        items: Sequence[tuple[str | None, bytes]],
        *,
        top_k: int | None = None,
    ) -> BatchPrediction:
        self.require_ready()
        if len(items) > self._settings.max_batch_size:
            raise PayloadTooLargeError(
                f"batch of {len(items)} exceeds the {self._settings.max_batch_size} image limit",
                details={"max_batch_size": self._settings.max_batch_size},
            )
        results: list[BatchItem] = []
        for index, (filename, data) in enumerate(items):
            try:
                prediction = self.predict_bytes(data, top_k=top_k)
            except ApiError as exc:
                results.append(
                    BatchItem(
                        index=index,
                        filename=filename,
                        error={"code": exc.code, "message": exc.message, "details": exc.details},
                    )
                )
                continue
            results.append(BatchItem(index=index, filename=filename, prediction=prediction))
        succeeded = sum(1 for item in results if item.prediction is not None)
        return BatchPrediction(
            count=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
            items=results,
        )
