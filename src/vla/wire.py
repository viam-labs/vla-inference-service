"""Wire codec for the protobuf Struct payloads exchanged between resources.

DoCommand carries a protobuf Struct, which holds only JSON types. Images become
base64 strings and float arrays become lists of lists.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


class WireError(ValueError):
    """Raised when a wire payload is malformed."""


def encode_image(img: np.ndarray, *, encoding: str = "jpeg", quality: int = 90) -> dict[str, Any]:
    if img.ndim != 3 or img.shape[2] != 3:
        raise WireError(f"expected HWC RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise WireError(f"expected uint8 image, got {img.dtype}")

    height, width, channels = img.shape

    if encoding == "raw":
        data = base64.b64encode(img.tobytes()).decode("ascii")
    elif encoding in ("jpeg", "png"):
        buf = io.BytesIO()
        pil_format = "JPEG" if encoding == "jpeg" else "PNG"
        kwargs = {"quality": quality} if encoding == "jpeg" else {}
        Image.fromarray(img).save(buf, format=pil_format, **kwargs)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
    else:
        raise WireError(f"unknown encoding {encoding!r}")

    return {
        "encoding": encoding,
        "data": data,
        "height": height,
        "width": width,
        "channels": channels,
    }


def decode_image(payload: dict[str, Any]) -> np.ndarray:
    encoding = payload.get("encoding")
    try:
        raw = base64.b64decode(payload["data"])
    except KeyError as exc:
        raise WireError("image payload missing 'data'") from exc

    if encoding == "raw":
        # Shape is carried explicitly so the decoder never has to infer it.
        try:
            shape = (int(payload["height"]), int(payload["width"]), int(payload["channels"]))
        except KeyError as exc:
            raise WireError("raw image payload missing shape fields") from exc
        expected = shape[0] * shape[1] * shape[2]
        if len(raw) != expected:
            raise WireError(f"raw image payload is {len(raw)} bytes, expected {expected}")
        # Every path below guarantees a writable array: a bare frombuffer
        # view (or a PIL-backed np.asarray) is read-only, and torch.from_numpy
        # warns on non-writable input downstream.
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()

    if encoding in ("jpeg", "png"):
        # np.asarray over a PIL image is read-only; copy for the same reason
        # the raw branch does — torch.from_numpy warns on non-writable input.
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)

    raise WireError(f"unknown encoding {encoding!r}")


def encode_matrix(m: np.ndarray) -> dict[str, Any]:
    if m.ndim != 2:
        raise WireError(f"expected 2D matrix, got shape {m.shape}")
    return {"rows": [[float(v) for v in row] for row in m]}


def decode_matrix(payload: dict[str, Any]) -> np.ndarray:
    rows = payload.get("rows")
    if rows is None:
        raise WireError("matrix payload missing 'rows'")
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise WireError(f"ragged matrix rows: widths {sorted(widths)}")
    return np.asarray(rows, dtype=np.float32)
