"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.errors import NotReadyError
from app.services.predictor import EmotionPredictor


def get_predictor(request: Request) -> EmotionPredictor:
    predictor: EmotionPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise NotReadyError("the service has not finished starting")
    return predictor


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


PredictorDep = Annotated[EmotionPredictor, Depends(get_predictor)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
