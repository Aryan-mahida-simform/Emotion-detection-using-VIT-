"""Model metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import PredictorDep, SettingsDep
from app.emotions import AFFECT_PROFILE, CANONICAL_EMOTIONS, LABEL_ALIASES
from app.schemas import LabelsResponse, ModelInfoResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse, summary="Active backend and its settings")
def model_info(
    request: Request,
    predictor: PredictorDep,
    settings: SettingsDep,
) -> ModelInfoResponse:
    descriptor = predictor.backend.descriptor
    return ModelInfoResponse(
        backend=descriptor.name,
        requested_backend=predictor.requested_backend,
        model_id=descriptor.model_id,
        device=descriptor.device,
        neural=descriptor.neural,
        fallback_reason=predictor.fallback_reason,
        labels=list(descriptor.labels),
        canonical_labels=list(CANONICAL_EMOTIONS),
        label_aliases=dict(LABEL_ALIASES),
        affect_profile=dict(AFFECT_PROFILE),
        image_size=settings.image_size,
        logit_temperature=settings.logit_temperature,
        default_top_k=settings.default_top_k,
        max_batch_size=settings.max_batch_size,
        max_image_bytes=settings.max_image_bytes,
        allowed_media_types=list(settings.accepts_media_type),
        upload_modes=list(getattr(request.app.state, "upload_modes", ())),
        warmup_latency_ms=predictor.warmup_latency_ms,
    )


@router.get("/labels", response_model=LabelsResponse, summary="Label vocabulary and affect anchors")
def model_labels() -> LabelsResponse:
    return LabelsResponse(
        labels=list(CANONICAL_EMOTIONS),
        canonical_labels=list(CANONICAL_EMOTIONS),
        label_aliases=dict(LABEL_ALIASES),
        affect_profile=dict(AFFECT_PROFILE),
    )
