"""Tests for image decoding and validation."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.imaging import check_media_type, check_size, decode_image, square_resize
from tests.conftest import encode_image


def test_decodes_png_to_rgb(png_bytes: bytes) -> None:
    image = decode_image(png_bytes)
    assert image.mode == "RGB"
    assert image.size == (96, 96)


def test_decodes_jpeg(jpeg_bytes: bytes) -> None:
    assert decode_image(jpeg_bytes).mode == "RGB"


@pytest.mark.parametrize(("mode", "fmt"), [("RGBA", "PNG"), ("L", "PNG"), ("P", "PNG"), ("CMYK", "JPEG")])
def test_converts_other_modes_to_rgb(mode: str, fmt: str) -> None:
    source = Image.new(mode, (32, 24))
    image = decode_image(encode_image(source, fmt))
    assert image.mode == "RGB"


def test_applies_exif_orientation(jpeg_bytes: bytes) -> None:
    source = Image.new("RGB", (40, 20), (180, 150, 120))
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).size == (20, 40)


def test_downscales_the_long_side() -> None:
    source = Image.new("RGB", (600, 300), (10, 20, 30))
    image = decode_image(encode_image(source), max_side=200)
    assert max(image.size) == 200
    assert image.size == (200, 100)


def test_rejects_unreadable_payloads() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"")
    with pytest.raises(InvalidImageError):
        decode_image(b"this is not an image")
    with pytest.raises(InvalidImageError):
        decode_image(encode_image(Image.new("RGB", (32, 32)))[:40])


def test_rejects_formats_outside_the_allow_list() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buffer, format="PPM")
    with pytest.raises(UnsupportedMediaTypeError):
        decode_image(buffer.getvalue())


def test_check_size_enforces_the_byte_budget() -> None:
    check_size(b"abc", 4)
    with pytest.raises(PayloadTooLargeError):
        check_size(b"abcde", 4)
    with pytest.raises(InvalidImageError):
        check_size(b"", 4)


def test_check_media_type_rejects_non_images() -> None:
    allowed = ("image/png", "image/jpeg")
    check_media_type("image/png", allowed)
    check_media_type(None, allowed)
    with pytest.raises(UnsupportedMediaTypeError) as error:
        check_media_type("text/plain", allowed)
    assert error.value.details["allowed"] == list(allowed)


def test_square_resize_crops_to_a_square() -> None:
    image = square_resize(Image.new("RGB", (500, 100), (5, 5, 5)), 64)
    assert image.size == (64, 64)
