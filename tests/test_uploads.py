"""Tests for reading upload bodies and base64 payloads."""

from __future__ import annotations

import base64
import io

import pytest

from app.errors import InvalidImageError, PayloadTooLargeError
from app.uploads import decode_base64_image, multipart_available, read_limited


def test_read_limited_returns_the_whole_stream() -> None:
    assert read_limited(io.BytesIO(b"abcdef"), 10) == b"abcdef"


def test_read_limited_stops_over_the_budget() -> None:
    with pytest.raises(PayloadTooLargeError) as error:
        read_limited(io.BytesIO(b"x" * 500), 128)
    assert error.value.details["max_bytes"] == 128


def test_read_limited_on_an_empty_stream() -> None:
    assert read_limited(io.BytesIO(b""), 10) == b""


def test_decode_base64_accepts_plain_and_data_url_payloads() -> None:
    payload = base64.b64encode(b"image-bytes").decode()
    assert decode_base64_image(payload) == b"image-bytes"
    assert decode_base64_image(f"data:image/png;base64,{payload}") == b"image-bytes"
    assert decode_base64_image(f"  {payload[:4]}\n{payload[4:]}  ") == b"image-bytes"


def test_decode_base64_rejects_bad_input() -> None:
    with pytest.raises(InvalidImageError):
        decode_base64_image("")
    with pytest.raises(InvalidImageError):
        decode_base64_image("not base64!!")


def test_multipart_availability_is_reported_as_a_bool() -> None:
    assert isinstance(multipart_available(), bool)
