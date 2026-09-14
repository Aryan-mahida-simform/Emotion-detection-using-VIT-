"""HTTP layer."""

from app.api.routes_health import router as health_router
from app.api.routes_model import router as model_router
from app.api.routes_predict import build_predict_router

__all__ = ["build_predict_router", "health_router", "model_router"]
