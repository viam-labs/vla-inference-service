import numpy as np
import pytest
from vla.wire import (
    encode_image, decode_image, encode_matrix, decode_matrix, WireError,
)


def test_jpeg_roundtrip_preserves_shape():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    payload = encode_image(img, encoding="jpeg", quality=90)
    out = decode_image(payload)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_raw_roundtrip_is_lossless():
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding="raw"))
    np.testing.assert_array_equal(out, img)


def test_raw_payload_carries_explicit_shape():
    img = np.zeros((5, 7, 3), dtype=np.uint8)
    payload = encode_image(img, encoding="raw")
    assert payload["height"] == 5
    assert payload["width"] == 7
    assert payload["channels"] == 3


def test_png_roundtrip_is_lossless():
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding="png"))
    np.testing.assert_array_equal(out, img)


def test_matrix_roundtrip():
    m = np.array([[1.5, -2.0], [0.0, 3.25]], dtype=np.float32)
    out = decode_matrix(encode_matrix(m))
    np.testing.assert_allclose(out, m)
    assert out.dtype == np.float32


def test_decode_matrix_rejects_ragged_rows():
    with pytest.raises(WireError, match="ragged"):
        decode_matrix({"rows": [[1.0, 2.0], [3.0]]})


def test_decode_image_rejects_unknown_encoding():
    with pytest.raises(WireError, match="encoding"):
        decode_image({"encoding": "webp", "data": ""})
