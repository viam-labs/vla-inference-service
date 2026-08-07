"""Wire codec for the protobuf Struct payloads exchanged between resources.

DoCommand carries a protobuf Struct, which holds only JSON types. Images become
base64 strings and float arrays become lists of lists.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

import numpy as np
from PIL import Image

from vla.config_util import VLAError

_IMAGE_ENCODINGS = ("raw", "jpeg", "png")


class WireError(VLAError, ValueError):
    """Raised when a wire payload is malformed."""


def encode_image(img: np.ndarray, *, encoding: str = "jpeg", quality: int = 90) -> dict[str, Any]:
    if img.ndim != 3 or img.shape[2] != 3:
        raise WireError(f"expected HWC RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise WireError(f"expected uint8 image, got {img.dtype}")
    if encoding not in _IMAGE_ENCODINGS:
        raise WireError(f"unknown encoding {encoding!r}")

    height, width, channels = img.shape

    if encoding == "raw":
        data = base64.b64encode(img.tobytes()).decode("ascii")
    else:
        # jpeg / png. libjpeg silently clamps an out-of-range quality to its
        # nearest valid value instead of erroring, which at 10 Hz can burn
        # several times the intended bandwidth with nothing in the log.
        if encoding == "jpeg" and (not isinstance(quality, int) or isinstance(quality, bool) or not (0 <= quality <= 100)):
            raise WireError(f"quality must be an integer in [0, 100], got {quality!r}")
        buf = io.BytesIO()
        pil_format = "JPEG" if encoding == "jpeg" else "PNG"
        kwargs = {"quality": quality} if encoding == "jpeg" else {}
        Image.fromarray(img).save(buf, format=pil_format, **kwargs)
        data = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "encoding": encoding,
        "data": data,
        "height": height,
        "width": width,
        "channels": channels,
    }


def decode_image(payload: dict[str, Any]) -> np.ndarray:
    # Check the encoding before touching 'data' at all: an unknown encoding
    # is a different fault than missing/corrupt data, and shouldn't be
    # masked by a base64 decode of a blob we're about to discard anyway.
    encoding = payload.get("encoding")
    if encoding not in _IMAGE_ENCODINGS:
        raise WireError(f"unknown encoding {encoding!r}")

    try:
        data = payload["data"]
    except KeyError as exc:
        raise WireError("image payload missing 'data'") from exc
    try:
        raw = base64.b64decode(data)
    except (binascii.Error, TypeError) as exc:
        raise WireError(f"invalid base64 'data' for {encoding!r} image") from exc

    if encoding == "raw":
        # Shape is carried explicitly so the decoder never has to infer it.
        try:
            height = int(payload["height"])
            width = int(payload["width"])
            channels = int(payload["channels"])
        except KeyError as exc:
            raise WireError("raw image payload missing shape fields") from exc
        except (TypeError, ValueError) as exc:
            raise WireError("raw image payload has non-numeric shape fields") from exc
        if height <= 0 or width <= 0:
            raise WireError(f"raw image payload has non-positive dimensions {(height, width)}")
        if channels != 3:
            raise WireError(f"raw image payload must have 3 channels, got {channels}")
        shape = (height, width, channels)
        expected = height * width * channels
        if len(raw) != expected:
            raise WireError(f"raw image payload is {len(raw)} bytes, expected {expected}")
        # Every path below guarantees a writable array: a bare frombuffer
        # view (or a PIL-backed np.asarray) is read-only, and torch.from_numpy
        # warns on non-writable input downstream.
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()

    # jpeg / png
    try:
        # np.array (not np.asarray) over a PIL image copies, for the same
        # reason the raw branch does: torch.from_numpy warns on non-writable
        # input.
        out = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise WireError(f"could not decode {encoding!r} image data") from exc

    # The encoder writes height/width on every path; honor them on every
    # path too. A silent mismatch here (e.g. a controller resize that didn't
    # happen) would otherwise surface many layers away as "the model
    # behaves oddly".
    declared_height = payload.get("height")
    declared_width = payload.get("width")
    if declared_height is not None and out.shape[0] != int(declared_height):
        raise WireError(
            f"decoded {encoding} image height {out.shape[0]} does not match declared height {declared_height}"
        )
    if declared_width is not None and out.shape[1] != int(declared_width):
        raise WireError(
            f"decoded {encoding} image width {out.shape[1]} does not match declared width {declared_width}"
        )
    return out


def encode_matrix(m: np.ndarray) -> dict[str, Any]:
    if m.ndim != 2:
        raise WireError(f"expected 2D matrix, got shape {m.shape}")
    return {"rows": m.astype(float, copy=False).tolist()}


def decode_matrix(payload: dict[str, Any]) -> np.ndarray:
    rows = payload.get("rows")
    if rows is None:
        raise WireError("matrix payload missing 'rows'")
    if not isinstance(rows, list):
        raise WireError(f"matrix payload 'rows' must be a list, got {type(rows).__name__}")
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            raise WireError(f"matrix payload row {i} must be a list, got {type(row).__name__}")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise WireError(f"ragged matrix rows: widths {sorted(widths)}")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if v is None:
                # Struct's null_value round-trips to None. Left unchecked
                # this becomes NaN below, and the safety layer's NaN
                # rejection then blames the model for a transport fault.
                raise WireError(f"matrix payload has a null value at row {i}, column {j}")
            if isinstance(v, (list, dict)):
                raise WireError(f"matrix payload value at row {i}, column {j} is not a scalar")
    return np.asarray(rows, dtype=np.float32)


def encode_vector(v: np.ndarray) -> dict[str, Any]:
    if v.ndim != 1:
        raise WireError(f"expected 1D vector, got shape {v.shape}")
    return {"values": v.astype(float, copy=False).tolist()}


def decode_vector(payload: dict[str, Any]) -> np.ndarray:
    values = payload.get("values")
    if values is None:
        raise WireError("vector payload missing 'values'")
    if not isinstance(values, list):
        raise WireError(f"vector payload 'values' must be a list, got {type(values).__name__}")
    for i, v in enumerate(values):
        if v is None:
            raise WireError(f"vector payload has a null value at index {i}")
        if isinstance(v, (list, dict)):
            raise WireError(f"vector payload value at index {i} is not a scalar")
    return np.asarray(values, dtype=np.float32)
