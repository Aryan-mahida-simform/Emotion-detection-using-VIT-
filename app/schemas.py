"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.emotions import CANONICAL_EMOTIONS


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class EmotionScore(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class Affect(BaseModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)


class ModelDescriptor(BaseModel):
    """Which model answered a request, so a client can log it per call."""

    backend: str
    model_id: str
    device: str
    neural: bool
    labels: list[str]


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: list[EmotionScore]
    probabilities: dict[str, float]
    affect: Affect
    model: ModelDescriptor
    latency_ms: float = Field(ge=0.0)


class BatchItem(BaseModel):
    index: int
    filename: str | None = None
    prediction: Prediction | None = None
    error: ErrorDetail | None = None


class BatchPrediction(BaseModel):
    count: int
    succeeded: int
    failed: int
    items: list[BatchItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    version: str
    uptime_seconds: float
    backend: str
    model_id: str
    upload_modes: list[str]


class ModelInfoResponse(BaseModel):
    backend: str
    requested_backend: str
    model_id: str
    device: str
    neural: bool
    fallback_reason: str | None
    labels: list[str]
    canonical_labels: list[str]
    label_aliases: dict[str, str]
    affect_profile: dict[str, tuple[float, float]]
    image_size: int
    logit_temperature: float
    default_top_k: int
    max_batch_size: int
    max_image_bytes: int
    allowed_media_types: list[str]
    upload_modes: list[str]
    warmup_latency_ms: float | None


class LabelsResponse(BaseModel):
    labels: list[str]
    canonical_labels: list[str]
    label_aliases: dict[str, str]
    affect_profile: dict[str, tuple[float, float]]


class Base64PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_base64: str = Field(min_length=1)
    filename: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=len(CANONICAL_EMOTIONS))


class RootResponse(BaseModel):
    name: str
    version: str
    docs_url: str
    endpoints: dict[str, str]
