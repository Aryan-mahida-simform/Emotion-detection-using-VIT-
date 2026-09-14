"""Request models and helpers for reading uploaded images."""

from __future__ import annotations

import base64
import binascii
from typing import BinaryIO

from fastapi import UploadFile

from app.errors import InvalidImageError, PayloadTooLargeError

CHUNK_SIZE = 64 * 1024


def multipart_available() -> bool:
    """True when FastAPI can build a File/Form endpoint on this machine."""

    try:
        from fastapi.dependencies.utils import ensure_multipart_is_installed
    except ImportError:
        return False
    try:
        ensure_multipart_is_installed()
    except RuntimeError:
        return False
    return True


def read_limited(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read a stream, refusing to buffer more than ``max_bytes``."""

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(
                f"image payload exceeds the {max_bytes} byte limit",
                details={"max_bytes": max_bytes},
            )
        chunks.append(chunk)
    return b"".join(chunks)


def read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    upload.file.seek(0)
    return read_limited(upload.file, max_bytes)


def decode_base64_image(payload: str) -> bytes:
    """Decode a base64 string, tolerating a data-URL prefix and whitespace."""

    text = "".join(payload.split())
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    if not text:
        raise InvalidImageError("image_base64 is empty after trimming whitespace")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"image_base64 is not valid base64: {exc}") from exc


def batch_labels(filenames: Sequence[str | None]) -> list[str]:
    return [name or f"item-{index}" for index, name in enumerate(filenames)]
