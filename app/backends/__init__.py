"""Classifier backends."""

from app.backends.base import BackendDescriptor, EmotionBackend
from app.backends.factory import BackendSelection, resolve_backend
from app.backends.reference import ReferenceBackend
from app.backends.vit import ViTBackend

__all__ = [
    "BackendDescriptor",
    "BackendSelection",
    "EmotionBackend",
    "ReferenceBackend",
    "ViTBackend",
    "resolve_backend",
]
