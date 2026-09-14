"""Health and readiness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app.errors import NotReadyError
from app.schemas import ErrorResponse, HealthResponse

router = APIRouter(tags=["health"])


def _snapshot(request: Request) -> HealthResponse:
    state = request.app.state
    settings = state.settings
    predictor = getattr(state, "predictor", None)
    ready = predictor is not None and predictor.is_ready
    started_at = getattr(state, "started_at", None)
    return HealthResponse(
        status="ok" if ready else "degraded",
        ready=ready,
        version=settings.version,
        uptime_seconds=round(time.monotonic() - started_at, 3) if started_at else 0.0,
        backend=predictor.backend.name if predictor else "unavailable",
        model_id=predictor.backend.descriptor.model_id if predictor else settings.model_id,
        upload_modes=list(getattr(state, "upload_modes", ())),
    )


@router.get("/health", response_model=HealthResponse, summary="Service liveness and active backend")
def health(request: Request) -> HealthResponse:
    return _snapshot(request)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "Backend not ready"}},
    summary="Readiness probe",
)
def readiness(request: Request) -> HealthResponse:
    snapshot = _snapshot(request)
    if not snapshot.ready:
        raise NotReadyError("the backend has not finished loading")
    return snapshot
