"""Tests for settings loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import (
    DEFAULT_MODEL_ID,
    Settings,
    load_settings,
    media_type_is_supported,
)


def test_defaults_are_usable_without_environment() -> None:
    settings = load_settings({})
    assert settings.backend == "auto"
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.image_size == 224
    assert settings.warmup_on_start is True
    assert settings.allow_model_download is False
    assert settings.cors_origins == ("*",)


def test_prefixed_variables_are_read_and_unknown_ones_ignored() -> None:
    settings = load_settings(
        {
            "EMOTION_API_BACKEND": "reference",
            "EMOTION_API_IMAGE_SIZE": "256",
            "EMOTION_API_LOGIT_TEMPERATURE": "0.5",
            "EMOTION_API_WARMUP_ON_START": "false",
            "EMOTION_API_MAX_IMAGE_BYTES": "4096",
            "OTHER_SERVICE_BACKEND": "vit",
            "EMOTION_API_NOT_A_FIELD": "1",
        }
    )
    assert settings.backend == "reference"
    assert settings.image_size == 256
    assert settings.logit_temperature == 0.5
    assert settings.warmup_on_start is False
    assert settings.max_image_bytes == 4096


def test_list_variables_accept_csv_and_json() -> None:
    csv = load_settings({"EMOTION_API_CORS_ORIGINS": "https://a.test, https://b.test"})
    assert csv.cors_origins == ("https://a.test", "https://b.test")
    js = load_settings({"EMOTION_API_CORS_ORIGINS": '["https://a.test"]'})
    assert js.cors_origins == ("https://a.test",)
    empty = load_settings({"EMOTION_API_CORS_ORIGINS": ""})
    assert empty.cors_origins == ()


def test_local_files_only_tracks_the_download_flag() -> None:
    assert load_settings({}).local_files_only is True
    assert load_settings({"EMOTION_API_ALLOW_MODEL_DOWNLOAD": "true"}).local_files_only is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "tensorflow"),
        ("device", "tpu"),
        ("log_level", "CHATTY"),
        ("image_size", 8),
        ("default_top_k", 0),
        ("logit_temperature", 0.0),
        ("max_batch_size", 0),
    ],
)
def test_invalid_settings_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_settings_are_frozen() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.backend = "reference"  # type: ignore[misc]


def test_media_type_accepts_blank_and_image_types() -> None:
    allowed = Settings().accepts_media_type
    assert media_type_is_supported(None, allowed)
    assert media_type_is_supported("", allowed)
    assert media_type_is_supported("application/octet-stream", allowed)
    assert media_type_is_supported("image/jpeg", allowed)
    assert media_type_is_supported("IMAGE/PNG; charset=binary", allowed)
    assert not media_type_is_supported("text/plain", allowed)
    assert not media_type_is_supported("application/json", allowed)
