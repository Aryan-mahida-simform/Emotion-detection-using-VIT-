"""FastAPI application factory and process entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import build_predict_router, health_router, model_router
from app.backends.base import EmotionBackend
from app.backends.factory import BackendSelection, resolve_backend
from app.config import Settings, load_settings
from app.errors import ApiError
from app.logging_config import configure_logging
from app.schemas import RootResponse
from app.services.predictor import EmotionPredictor

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    backend: EmotionBackend | None = None,
) -> FastAPI:
    """Build the application.

    ``backend`` bypasses backend resolution, which lets tests drive the API
    with a classifier whose scores they control.
    """

    resolved_settings = settings or load_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if backend is not None:
            selection = BackendSelection(backend=backend, requested=backend.name)
        else:
            selection = resolve_backend(resolved_settings)
            if selection.is_fallback:
                logger.warning("running on the reference backend: %s", selection.fallback_reason)

        predictor = EmotionPredictor(selection.backend, resolved_settings, selection)
        if resolved_settings.warmup_on_start:
            latency = predictor.warmup()
            if latency is not None:
                logger.info("warmup took %.3f ms", latency)
        application.state.predictor = predictor
        application.state.started_at = time.monotonic()
        logger.info(
            "ready: backend=%s model=%s labels=%s",
            selection.backend.name,
            selection.backend.descriptor.model_id,
            selection.backend.labels,
        )
        try:
            yield
        finally:
            application.state.predictor = None
            selection.backend.close()

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.version,
        summary="Vision Transformer emotion classification with a labelled fallback backend.",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings

    predict_router, multipart_enabled = build_predict_router()
    modes = ["base64", "raw"]
    if multipart_enabled:
        modes.insert(0, "multipart")
    application.state.upload_modes = modes

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    application.include_router(health_router)
    application.include_router(model_router)
    application.include_router(predict_router)

    @application.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(exc.to_payload()))

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "the request did not match the endpoint schema",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "the server hit an unexpected error",
                    "details": {},
                }
            },
        )

    @application.get("/", response_model=RootResponse, summary="Service banner")
    def root() -> RootResponse:
        endpoints: dict[str, Any] = {
            "health": "/health",
            "readiness": "/health/ready",
            "model_info": "/model/info",
            "model_labels": "/model/labels",
            "predict_upload": "/predict",
            "predict_raw": "/predict/raw",
            "predict_base64": "/predict/base64",
            "predict_batch": "/predict/batch",
            "openapi": "/openapi.json",
            "docs": "/docs",
        }
        return RootResponse(
            name=resolved_settings.app_name,
            version=resolved_settings.version,
            docs_url="/docs",
            endpoints=endpoints,
        )

    return application


app = create_app()


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
