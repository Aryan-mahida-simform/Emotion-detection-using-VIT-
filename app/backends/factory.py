"""Pick the backend the service runs with."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.backends.base import EmotionBackend
from app.backends.reference import ReferenceBackend
from app.backends.vit import ViTBackend
from app.config import Settings
from app.errors import ApiError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendSelection:
    backend: EmotionBackend
    requested: str
    fallback_reason: str | None = None

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None


def resolve_backend(settings: Settings) -> BackendSelection:
    """Return the backend for these settings.

    ``reference`` never loads a checkpoint. ``vit`` fails loudly when the
    checkpoint is missing, because the operator asked for it by name. ``auto``
    tries the checkpoint and keeps the reference scorer with a recorded reason
    when that fails.
    """

    if settings.backend == "reference":
        return BackendSelection(backend=ReferenceBackend(), requested="reference")

    available, reason = ViTBackend.dependencies_available()
    if settings.backend == "vit":
        if not available:
            raise ConfigurationError(
                f"backend=vit was requested but {reason}",
                details={"model_id": settings.model_id},
            )
        backend = _build_vit(settings)
        backend.load()
        return BackendSelection(backend=backend, requested="vit")

    if not available:
        logger.warning("falling back to the reference backend: %s", reason)
        return BackendSelection(backend=ReferenceBackend(), requested="auto", fallback_reason=reason)

    try:
        backend = _build_vit(settings)
        backend.load()
    except ApiError as exc:
        logger.warning("falling back to the reference backend: %s", exc.message)
        return BackendSelection(
            backend=ReferenceBackend(),
            requested="auto",
            fallback_reason=exc.message,
        )
    return BackendSelection(backend=backend, requested="auto")


def _build_vit(settings: Settings) -> ViTBackend:
    return ViTBackend(
        settings.model_id,
        device=settings.device,
        local_files_only=settings.local_files_only,
    )
