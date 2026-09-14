"""Prediction endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, File, Header, Query, UploadFile

from app.api.deps import PredictorDep, SettingsDep
from app.emotions import CANONICAL_EMOTIONS
from app.schemas import Base64PredictionRequest, BatchPrediction, ErrorResponse, Prediction
from app.uploads import decode_base64_image, multipart_available, read_upload

logger = logging.getLogger(__name__)

TopK = Annotated[
    int | None,
    Query(ge=1, le=len(CANONICAL_EMOTIONS), description="How many ranked classes to return"),
]

DeclaredContentType = Annotated[
    str | None,
    Header(
        alias="content-type",
        description="Declared image type. An unlabelled body is sniffed instead.",
    ),
]

CLIENT_ERRORS = {
    413: {"model": ErrorResponse, "description": "Image or batch too large"},
    415: {"model": ErrorResponse, "description": "Content type is not an accepted image type"},
    422: {"model": ErrorResponse, "description": "Payload is not a decodable image"},
}


def build_predict_router() -> tuple[APIRouter, bool]:
    """Build the prediction routes.

    When FastAPI reports that no multipart parser is installed, ``/predict``
    takes the raw image bytes instead of a file part, so the endpoint never
    disappears.
    """

    router = APIRouter(tags=["predict"])
    multipart_enabled = multipart_available()

    if multipart_enabled:

        @router.post(
            "/predict",
            response_model=Prediction,
            responses=CLIENT_ERRORS,
            summary="Classify one uploaded image",
        )
        def predict_upload(
            predictor: PredictorDep,
            settings: SettingsDep,
            file: Annotated[UploadFile, File(description="JPEG, PNG, WEBP, BMP, GIF or TIFF")],
            top_k: TopK = None,
        ) -> Prediction:
            data = read_upload(file, settings.max_image_bytes)
            return predictor.predict_bytes(data, media_type=file.content_type, top_k=top_k)

    else:

        @router.post(
            "/predict",
            response_model=Prediction,
            responses=CLIENT_ERRORS,
            summary="Classify raw image bytes (no multipart parser installed)",
        )
        def predict_body(
            predictor: PredictorDep,
            payload: bytes = Body(media_type="application/octet-stream"),
            content_type: DeclaredContentType = None,
            top_k: TopK = None,
        ) -> Prediction:
            return predictor.predict_bytes(payload, media_type=content_type, top_k=top_k)

    @router.post(
        "/predict/raw",
        response_model=Prediction,
        responses=CLIENT_ERRORS,
        summary="Classify raw image bytes sent as the request body",
    )
    def predict_raw(
        predictor: PredictorDep,
        payload: bytes = Body(media_type="application/octet-stream"),
        content_type: DeclaredContentType = None,
        top_k: TopK = None,
    ) -> Prediction:
        return predictor.predict_bytes(payload, media_type=content_type, top_k=top_k)

    @router.post(
        "/predict/base64",
        response_model=Prediction,
        responses=CLIENT_ERRORS,
        summary="Classify an image provided as a base64 string",
    )
    def predict_base64(
        predictor: PredictorDep,
        request: Base64PredictionRequest,
    ) -> Prediction:
        data = decode_base64_image(request.image_base64)
        return predictor.predict_bytes(data, media_type=None, top_k=request.top_k)

    @router.post(
        "/predict/batch",
        response_model=BatchPrediction,
        responses=CLIENT_ERRORS,
        summary="Classify several images; per-image failures are reported in place",
    )
    def predict_batch(
        predictor: PredictorDep,
        settings: SettingsDep,
        files: Annotated[list[UploadFile], File(description="Two or more images")],
        top_k: TopK = None,
    ) -> BatchPrediction:
        if not files:
            return BatchPrediction(count=0, succeeded=0, failed=0, items=[])
        items = [
            (upload.filename, read_upload(upload, settings.max_image_bytes)) for upload in files
        ]
        return predictor.predict_batch(items, top_k=top_k)

    return router, multipart_enabled
