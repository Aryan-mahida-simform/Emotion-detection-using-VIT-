"""Image decoding and normalisation for the prediction endpoints."""

from __future__ import annotations

import io
from typing import Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import media_type_is_supported
from app.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaTypeError

DECODABLE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"})
MAX_PIXELS = 40_000_000

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def check_media_type(media_type: str | None, allowed: Sequence[str]) -> None:
    if not media_type_is_supported(media_type, allowed):
        raise UnsupportedMediaTypeError(
            f"content type {media_type!r} is not accepted",
            details={"allowed": list(allowed)},
        )


def check_size(data: bytes, max_bytes: int) -> None:
    if not data:
        raise InvalidImageError("image payload is empty")
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"image payload of {len(data)} bytes exceeds the {max_bytes} byte limit",
            details={"size_bytes": len(data), "max_bytes": max_bytes},
        )


def decode_image(data: bytes, *, max_side: int | None = None) -> Image.Image:
    """Decode bytes into an upright RGB image.

    The payload is verified first, so a truncated file fails as a client error
    rather than as a partial decode further down the pipeline.
    """

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            probe.verify()
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"image dimensions exceed the {MAX_PIXELS} pixel cap") from exc
    except UnidentifiedImageError as exc:
        raise InvalidImageError("payload is not a readable image") from exc
    except OSError as exc:
        raise InvalidImageError(f"payload is not a readable image: {exc}") from exc

    if fmt not in DECODABLE_FORMATS:
        raise UnsupportedMediaTypeError(
            f"image format {fmt or 'unknown'} is not accepted",
            details={"allowed_formats": sorted(DECODABLE_FORMATS)},
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
    except Image.DecompressionBombError as exc:
        raise PayloadTooLargeError(f"image dimensions exceed the {MAX_PIXELS} pixel cap") from exc
    except OSError as exc:
        raise InvalidImageError(f"image could not be decoded: {exc}") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if image.width == 0 or image.height == 0:
        raise InvalidImageError("image has zero pixels")
    return image


def square_resize(image: Image.Image, size: int) -> Image.Image:
    """Resize to a fixed square, cropping the longer side from the centre."""

    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
