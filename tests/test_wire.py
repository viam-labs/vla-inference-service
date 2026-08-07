import base64

import numpy as np
import pytest
from viam.utils import dict_to_struct, struct_to_dict
from vla.config_util import VLAError
from vla.wire import (
    encode_image, decode_image, encode_matrix, decode_matrix,
    encode_vector, decode_vector, WireError,
)


def test_wire_error_is_a_vla_error():
    assert issubclass(WireError, VLAError)


def test_wire_error_is_still_a_value_error():
    assert issubclass(WireError, ValueError)


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


# --- item 1: null entries must not silently become NaN -----------------------


def test_decode_matrix_rejects_null_value():
    with pytest.raises(WireError, match="row 0, column 1"):
        decode_matrix({"rows": [[1.0, None]]})


# --- item 2: compressed-path failures must surface as WireError --------------


def test_decode_image_wraps_invalid_base64():
    with pytest.raises(WireError, match="jpeg"):
        decode_image({"encoding": "jpeg", "data": "abc"})


def test_decode_image_wraps_non_string_data():
    with pytest.raises(WireError, match="jpeg"):
        decode_image({"encoding": "jpeg", "data": None})


def test_decode_image_wraps_corrupt_compressed_data():
    garbage = base64.b64encode(b"not an image").decode("ascii")
    with pytest.raises(WireError, match="jpeg"):
        decode_image({"encoding": "jpeg", "data": garbage})


def test_decode_image_wraps_truncated_compressed_data():
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    payload = encode_image(img, encoding="jpeg")
    raw = base64.b64decode(payload["data"])
    payload["data"] = base64.b64encode(raw[: len(raw) // 2]).decode("ascii")
    with pytest.raises(WireError, match="jpeg"):
        decode_image(payload)


def test_decode_image_wraps_non_numeric_raw_shape():
    payload = encode_image(np.zeros((4, 4, 3), dtype=np.uint8), encoding="raw")
    payload["height"] = "not-a-number"
    with pytest.raises(WireError, match="raw"):
        decode_image(payload)


# --- item 3: compressed path must honor the declared shape --------------------


def test_decode_image_rejects_shape_mismatch_on_compressed_path():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    payload = encode_image(img, encoding="png")
    payload["height"] = 999
    with pytest.raises(WireError, match="999"):
        decode_image(payload)


# --- item 4: decode_matrix must reject non-matrix input -----------------------


def test_decode_matrix_rejects_flat_list():
    with pytest.raises(WireError):
        decode_matrix({"rows": [1.0, 2.0]})


def test_decode_matrix_rejects_non_list_rows():
    with pytest.raises(WireError):
        decode_matrix({"rows": 5})


def test_decode_matrix_rejects_nested_rows():
    with pytest.raises(WireError):
        decode_matrix({"rows": [[[1, 2]], [[3, 4]]]})


# --- item 6: quality must be validated, not silently clamped ------------------


def test_encode_image_rejects_quality_out_of_range():
    with pytest.raises(WireError, match="quality"):
        encode_image(np.zeros((4, 4, 3), dtype=np.uint8), encoding="jpeg", quality=900)


def test_encode_image_rejects_non_integer_quality():
    with pytest.raises(WireError, match="quality"):
        encode_image(np.zeros((4, 4, 3), dtype=np.uint8), encoding="jpeg", quality=90.5)


# --- item 7: encoding must be checked before the data is touched --------------


def test_decode_image_rejects_unknown_encoding_before_touching_data():
    with pytest.raises(WireError, match="encoding"):
        decode_image({"encoding": "webp"})


# --- item 8: raw dimensions must be positive with channels == 3 ---------------


def test_decode_image_rejects_wrong_channel_count_on_raw_path():
    payload = encode_image(np.zeros((4, 4, 3), dtype=np.uint8), encoding="raw")
    # Byte count coincidentally matches a single-channel interpretation of a
    # differently-shaped buffer, so this can only be caught by validating
    # channels explicitly.
    payload["channels"] = 1
    payload["width"] = 12
    with pytest.raises(WireError, match="channels"):
        decode_image(payload)


def test_decode_image_rejects_non_positive_raw_dimensions():
    payload = encode_image(np.zeros((4, 4, 3), dtype=np.uint8), encoding="raw")
    payload["height"] = 0
    with pytest.raises(WireError, match="dimensions"):
        decode_image(payload)


# --- item 5: own the 1D state-vector boundary too ------------------------------


def test_vector_roundtrip():
    v = np.array([1.0, -2.5, 3.0], dtype=np.float32)
    out = decode_vector(encode_vector(v))
    np.testing.assert_allclose(out, v)
    assert out.dtype == np.float32


def test_encode_vector_emits_json_native_floats():
    payload = encode_vector(np.array([1.5, -2.0], dtype=np.float32))
    assert all(type(v) is float for v in payload["values"])


def test_encode_vector_rejects_non_1d_input():
    with pytest.raises(WireError, match="1D"):
        encode_vector(np.zeros((2, 2), dtype=np.float32))


def test_decode_vector_rejects_missing_values():
    with pytest.raises(WireError, match="values"):
        decode_vector({})


def test_decode_vector_rejects_non_list_values():
    with pytest.raises(WireError, match="values"):
        decode_vector({"values": 5})


def test_decode_vector_rejects_null_entry():
    with pytest.raises(WireError, match="index 1"):
        decode_vector({"values": [1.0, None]})


# --- item 9: the constraint this module exists for, exercised for real --------


def test_payloads_survive_a_real_protobuf_struct():
    # This module exists because Struct carries only JSON types. Assert it,
    # rather than assuming our dicts are Struct-safe.
    img = np.random.randint(0, 255, (8, 4, 3), dtype=np.uint8)
    payload = {
        # png (lossless), not the jpeg default: this test isolates
        # Struct-safety of the payload shape, not jpeg's lossy compression,
        # which would fail this pixel-exact comparison on its own.
        "images": {"observation.images.top": encode_image(img, encoding="png")},
        "actions": encode_matrix(np.zeros((3, 6), dtype=np.float32)),
        "state": encode_vector(np.zeros(6, dtype=np.float32)),
    }
    out = struct_to_dict(dict_to_struct(payload))
    np.testing.assert_array_equal(decode_image(out["images"]["observation.images.top"]), img)
    assert decode_matrix(out["actions"]).shape == (3, 6)
    assert decode_vector(out["state"]).shape == (6,)
