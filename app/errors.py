"""Error types that map onto HTTP responses."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"error": error}


class InvalidImageError(ApiError):
    status_code = 422
    code = "invalid_image"


class UnsupportedMediaTypeError(ApiError):
    status_code = 415
    code = "unsupported_media_type"


class PayloadTooLargeError(ApiError):
    status_code = 413
    code = "payload_too_large"


class BackendUnavailableError(ApiError):
    status_code = 503
    code = "backend_unavailable"


class ModelOutputError(ApiError):
    status_code = 502
    code = "model_output_invalid"


class NotReadyError(ApiError):
    status_code = 503
    code = "not_ready"


class ConfigurationError(ApiError):
    status_code = 500
    code = "configuration_error"
