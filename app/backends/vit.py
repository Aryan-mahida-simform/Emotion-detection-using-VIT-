"""Vision Transformer classifier loaded through transformers.

torch and transformers are imported inside :meth:`load`, so the rest of the
service runs when the machine-learning stack is absent.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from PIL import Image

from app.backends.base import BackendDescriptor, EmotionBackend
from app.errors import BackendUnavailableError, ConfigurationError, ModelOutputError
from app.emotions import labels_from_id2label

logger = logging.getLogger(__name__)

VIT_BACKEND_NAME = "vit"


class ViTBackend(EmotionBackend):
    """Wraps a ``ViTForImageClassification`` checkpoint from the Hugging Face hub."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.local_files_only = local_files_only
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._descriptor = BackendDescriptor(
            name=VIT_BACKEND_NAME,
            model_id=model_id,
            device=device,
            neural=True,
            labels=[],
        )

    @staticmethod
    def dependencies_available() -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            return False, f"torch/transformers unavailable: {exc}"
        return True, ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_ready(self) -> bool:
        return self.is_loaded

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except Exception as exc:
            raise BackendUnavailableError(
                f"torch/transformers unavailable: {exc}",
                details={"model_id": self.model_id},
            ) from exc

        device = self._resolve_device(torch)
        try:
            processor = AutoImageProcessor.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
            model = AutoModelForImageClassification.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
        except Exception as exc:
            hint = (
                "weights are not in the local cache and EMOTION_API_ALLOW_MODEL_DOWNLOAD is false"
                if self.local_files_only
                else "check the model id and the network connection"
            )
            raise BackendUnavailableError(
                f"could not load {self.model_id}: {exc}",
                details={"model_id": self.model_id, "hint": hint},
            ) from exc

        model.to(device)
        model.eval()
        config = model.config
        num_labels = int(getattr(config, "num_labels", 0) or 0)
        id2label = getattr(config, "id2label", None)
        if num_labels <= 0:
            num_labels = len(id2label) if id2label else 0
        if num_labels <= 0:
            raise BackendUnavailableError(f"{self.model_id} does not declare any output labels")
        labels = labels_from_id2label(id2label, num_labels)

        self._processor = processor
        self._model = model
        self._torch = torch
        self._descriptor = BackendDescriptor(
            name=VIT_BACKEND_NAME,
            model_id=self.model_id,
            device=device,
            neural=True,
            labels=labels,
        )
        logger.info("loaded %s on %s with labels %s", self.model_id, device, labels)

    def _resolve_device(self, torch: Any) -> str:
        if self.requested_device == "cpu":
            return "cpu"
        if self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise ConfigurationError("device=cuda was requested but CUDA is not available")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, images: Sequence[Image.Image]) -> list[list[float]]:
        if self._model is None or self._processor is None:
            raise BackendUnavailableError(f"{self.model_id} has not been loaded")
        if not images:
            return []

        with self._lock:
            inputs = self._processor(images=list(images), return_tensors="pt")
            device_inputs = {
                key: value.to(self._descriptor.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with self._torch.inference_mode():
                logits = self._model(**device_inputs).logits
            rows = logits.detach().to("cpu").tolist()

        expected = len(self.labels)
        for row in rows:
            if len(row) != expected:
                raise ModelOutputError(
                    f"{self.model_id} returned {len(row)} scores for {expected} labels"
                )
        return rows

    def close(self) -> None:
        self._model = None
        self._processor = None
