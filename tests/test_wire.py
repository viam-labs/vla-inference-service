import numpy as np
import pytest
from vla.wire import (
    encode_image, decode_image, encode_matrix, decode_matrix, WireError,
)


def test_jpeg_roundtrip_preserves_shape():
    img = np.random.randint(0, 255, (224, 160, 3), dtype=np.uint8)
    payload = encode_image(img, encoding="jpeg", quality=90)
    out = decode_image(payload)
    assert out.shape == (224, 160, 3)
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


@pytest.mark.parametrize("encoding", ["raw", "jpeg", "png"])
def test_decode_image_returns_writable_array(encoding):
    # torch.from_numpy warns on read-only input; every path must be writable.
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding=encoding))
    assert out.flags.writeable
    out[0, 0, 0] = 42  # must not raise


def test_raw_decode_uses_payload_shape_not_a_guess():
    # Deliberately a different shape from every other fixture: a decoder that
    # hardcodes or infers dimensions passes the other tests but fails this one.
    img = np.random.randint(0, 255, (3, 11, 3), dtype=np.uint8)
    out = decode_image(encode_image(img, encoding="raw"))
    assert out.shape == (3, 11, 3)
    np.testing.assert_array_equal(out, img)


def test_encode_matrix_emits_json_native_floats():
    # This is the module's whole purpose: numpy scalars are not JSON types and
    # would fail protobuf Struct serialization at runtime.
    payload = encode_matrix(np.array([[1.5, -2.0]], dtype=np.float32))
    assert all(type(v) is float for row in payload["rows"] for v in row)


def test_encode_image_rejects_non_uint8():
    with pytest.raises(WireError, match="uint8"):
        encode_image(np.zeros((4, 4, 3), dtype=np.float32))


def test_encode_image_rejects_wrong_channel_count():
    with pytest.raises(WireError, match="HWC RGB"):
        encode_image(np.zeros((4, 4, 4), dtype=np.uint8))


def test_encode_image_rejects_2d_input():
    with pytest.raises(WireError, match="HWC RGB"):
        encode_image(np.zeros((4, 4), dtype=np.uint8))


def test_decode_image_rejects_truncated_raw_payload():
    payload = encode_image(np.zeros((4, 4, 3), dtype=np.uint8), encoding="raw")
    payload["height"] = 8  # claims more bytes than were sent
    with pytest.raises(WireError, match="bytes"):
        decode_image(payload)


def test_decode_image_missing_data_raises_wire_error():
    with pytest.raises(WireError, match="data"):
        decode_image({"encoding": "jpeg"})


def test_decode_matrix_empty_rows_returns_empty_2d():
    out = decode_matrix({"rows": []})
    assert out.shape == (0, 0)
    assert out.dtype == np.float32


def test_decode_matrix_missing_rows_raises_wire_error():
    with pytest.raises(WireError, match="rows"):
        decode_matrix({})
