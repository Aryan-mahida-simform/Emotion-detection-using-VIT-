"""Runtime configuration, read from EMOTION_API_* environment variables."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import __version__
from app.emotions import CANONICAL_EMOTIONS

ENV_PREFIX = "EMOTION_API_"
SUPPORTED_BACKENDS: tuple[str, ...] = ("auto", "vit", "reference")
SUPPORTED_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

DEFAULT_MODEL_ID = "dima806/facial_emotions_image_detection"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_ALLOWED_MEDIA_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
    "image/tiff",
)


class Settings(BaseModel):
    """Validated service settings.

    ``backend="auto"`` prefers the ViT checkpoint and falls back to the
    reference scorer when the checkpoint cannot be loaded. Naming
    ``backend="vit"`` makes a missing checkpoint a startup error instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_name: str = "Emotion detection API"
    version: str = __version__
    backend: str = "auto"
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    image_size: int = Field(default=DEFAULT_IMAGE_SIZE, ge=32, le=1024)
    logit_temperature: float = Field(default=1.0, gt=0.0, le=100.0)
    default_top_k: int = Field(default=3, ge=1, le=len(CANONICAL_EMOTIONS))
    max_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    max_image_side: int = Field(default=2048, ge=64)
    max_batch_size: int = Field(default=8, ge=1, le=64)
    warmup_on_start: bool = True
    allow_model_download: bool = False
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ("*",)
    allowed_media_types: tuple[str, ...] = DEFAULT_ALLOWED_MEDIA_TYPES
    static_dir: str | None = None

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {value!r}")
        return normalized

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_DEVICES:
            raise ValueError(f"device must be one of {SUPPORTED_DEVICES}, got {value!r}")
        return normalized

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"log_level must be a standard level name, got {value!r}")
        return normalized

    @field_validator("cors_origins", "allowed_media_types", mode="before")
    @classmethod
    def _split_list(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON list")
            return tuple(str(item) for item in parsed)
        return tuple(part.strip() for part in text.split(",") if part.strip())

    @property
    def local_files_only(self) -> bool:
        return not self.allow_model_download

    @property
    def accepts_media_type(self) -> tuple[str, ...]:
        return tuple(media.lower() for media in self.allowed_media_types)


def _coerce(name: str, value: str) -> Any:
    if name.startswith("max_") or name in {"image_size", "default_top_k"}:
        return int(value)
    if name in {"logit_temperature",}:
        return float(value)
    return value


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from ``EMOTION_API_<FIELD>`` variables.

    Unknown variables are ignored, so a shared .env file can carry settings for
    several services.
    """

    source = os.environ if environ is None else environ
    known = set(Settings.model_fields)
    values: dict[str, Any] = {}
    for key, raw in source.items():
        if not key.startswith(ENV_PREFIX):
            continue
        field = key[len(ENV_PREFIX) :].lower()
        if field not in known:
            continue
        values[field] = _coerce(field, raw)
    return Settings(**values)


def media_type_is_supported(media_type: str | None, allowed: Sequence[str]) -> bool:
    """Accept an explicit image type, or an unlabelled payload we can sniff."""

    if media_type is None:
        return True
    normalized = media_type.split(";", 1)[0].strip().lower()
    if not normalized or normalized == "application/octet-stream":
        return True
    return normalized in {item.lower() for item in allowed}
