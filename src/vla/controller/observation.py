"""Assemble one policy observation from Viam cameras and the arm.

Camera reads use the real installed-sdk API, `get_images()` (plural),
returning `(Sequence[NamedImage], ResponseMetadata)` -- not the `get_image()`
singular an earlier plan draft assumed, which does not exist on viam-sdk
0.80.0.

Reads are gathered, not sequential: at 10 Hz the whole tick has a 100 ms
budget, and two serial camera reads alone can consume most of it. A camera
failure fails the whole tick -- this module never substitutes a black frame
or reuses a stale one; both would silently corrupt policy input in ways that
look like bad model behavior rather than a plumbing fault.

Two independent staleness signals are tracked, because they catch different
faults:
  - `duration_s` / `DEFAULT_DURATION_WARN_S`: how long *this* assembly took.
    Catches a slow read (network camera, contention).
  - `metadata.captured_at` / `STALE_FRAME_WARN_S`: how old the frame *was*
    when the camera returned it. Catches a camera silently serving a
    buffered stale frame, which a fast-but-wrong read cannot reveal via
    duration alone.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from vla.config_util import VLAError
from vla.wire import WireError, encode_image

from .gripper import GripperAdapter
from .units import from_degrees

LOGGER = logging.getLogger(__name__)

# How long an observation assembly may take before it is worth a warning.
# 0.1s (100ms) is the whole tick budget at the project's target 10 Hz --
# past this, the control loop is already falling behind regardless of why.
DEFAULT_DURATION_WARN_S = 0.1

# How old a frame's `captured_at` may be before it is worth a warning. Set
# above the duration budget: an assembly that itself takes close to the
# duration budget will naturally produce a captured_at a similar age, and
# this constant should flag a camera serving a *buffered* frame, not merely
# echo the duration warning under a different name.
STALE_FRAME_WARN_S = 0.5


class ObservationError(VLAError, RuntimeError):
    """Raised when an observation cannot be assembled.

    A `RuntimeError` base (matching `SafetyError`'s convention in
    `safety.py`): most failures here are runtime conditions (a camera read
    failing, an arm read failing) rather than a malformed argument.
    """


@dataclass(frozen=True)
class Observation:
    images: dict[str, dict[str, Any]]
    state: np.ndarray
    duration_s: float


class ObservationBuilder:
    def __init__(
        self,
        *,
        cameras: Mapping[str, Any],
        arm: Any,
        gripper: GripperAdapter,
        state_joint_indices: Sequence[int],
        state_units: str,
        image_sizes: Mapping[str, tuple[int, int]],
        image_encoding: str = "jpeg",
        jpeg_quality: int = 90,
        image_fit: str = "pad",
        duration_warn_s: float = DEFAULT_DURATION_WARN_S,
        stale_frame_warn_s: float = STALE_FRAME_WARN_S,
    ) -> None:
        self._cameras = dict(cameras)
        self._arm = arm
        self._gripper = gripper
        self._indices = list(state_joint_indices)
        self._units = state_units
        self._sizes = dict(image_sizes)
        self._encoding = image_encoding
        self._quality = jpeg_quality
        self._fit = image_fit
        self._duration_warn_s = duration_warn_s
        self._stale_frame_warn_s = stale_frame_warn_s

    async def build(self) -> Observation:
        started = time.perf_counter()
        keys = list(self._cameras)

        # Gathered, not sequential: at 10 Hz the whole tick has a 100 ms
        # budget and two serial camera reads can consume most of it.
        tasks: list[Any] = [self._read_camera(key) for key in keys]
        tasks.append(self._arm.get_joint_positions())
        needs_gripper_read = self._gripper.has_normalized_tail
        if needs_gripper_read:
            tasks.append(self._gripper.read())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        gripper_value: Any = None
        if needs_gripper_read:
            gripper_value = results[-1]
            results = results[:-1]
        joints = results[-1]
        frame_results = results[: len(keys)]

        for key, result in zip(keys, frame_results):
            self._raise_if_failed(result, f"camera {key!r} read failed")
        self._raise_if_failed(joints, "arm joint read failed")
        if needs_gripper_read:
            self._raise_if_failed(gripper_value, "gripper read failed")

        images = {
            key: self._encode(key, arr) for key, (arr, _age) in zip(keys, frame_results)
        }
        for key, (_arr, age) in zip(keys, frame_results):
            if age is not None and age > self._stale_frame_warn_s:
                LOGGER.warning(
                    "camera %r frame is stale: %.3fs old, exceeding the %.3fs staleness "
                    "threshold; the camera may be serving a buffered frame",
                    key,
                    age,
                    self._stale_frame_warn_s,
                )

        state = self._build_state(list(joints.values), gripper_value)

        duration = time.perf_counter() - started
        if duration > self._duration_warn_s:
            LOGGER.warning(
                "observation assembly took %.3fs, exceeding the %.3fs tick budget; "
                "check camera/arm read latency",
                duration,
                self._duration_warn_s,
            )
        return Observation(images=images, state=state, duration_s=duration)

    @staticmethod
    def _raise_if_failed(result: Any, message: str) -> None:
        if isinstance(result, ObservationError):
            raise result
        if isinstance(result, Exception):
            raise ObservationError(f"{message}: {result}") from result

    async def _read_camera(self, key: str) -> tuple[np.ndarray, float | None]:
        """Read one camera and return `(rgb_array, frame_age_s_or_None)`.

        Raises `ObservationError` naming `key` when the camera returns zero
        images -- `get_images()` returning an empty sequence must never
        surface as a bare `IndexError` from `images[0]`.
        """
        cam = self._cameras[key]
        images, metadata = await cam.get_images()
        if not images:
            raise ObservationError(f"camera {key!r} returned zero images")
        arr = self._to_array(images[0])
        age = self._frame_age_s(metadata)
        return arr, age

    @staticmethod
    def _frame_age_s(metadata: Any) -> float | None:
        """True frame age from `ResponseMetadata.captured_at`, or `None`.

        `None` covers both "no metadata object" and a driver that never
        populated `captured_at`, which protobuf leaves at its zero-value
        (1970-01-01) default -- that must not be misread as an ancient
        frame.
        """
        captured_at = getattr(metadata, "captured_at", None)
        if captured_at is None:
            return None
        if captured_at.seconds == 0 and captured_at.nanos == 0:
            return None
        captured_epoch_s = captured_at.ToNanoseconds() / 1e9
        return time.time() - captured_epoch_s

    @staticmethod
    def _to_array(frame: Any) -> np.ndarray:
        from viam.media.utils.pil import viam_to_pil_image

        return np.asarray(viam_to_pil_image(frame).convert("RGB"), dtype=np.uint8)

    def _encode(self, key: str, arr: np.ndarray) -> dict[str, Any]:
        size = self._sizes.get(key)
        if size is None:
            # Never forward an unresized frame. A 1080p frame under
            # image_encoding="raw" base64-encodes to ~8.3 MB, which exceeds
            # typical gRPC message limits -- and the policy would receive a
            # resolution it was not trained on either way.
            raise ObservationError(
                f"no expected size for {key!r}; policy specs did not declare it"
            )
        target_h, target_w = int(size[0]), int(size[1])
        if arr.shape[:2] != (target_h, target_w):
            if self._fit == "stretch":
                # The original behavior, kept as the explicit non-default
                # choice: a plain resize onto the declared shape, distorting
                # a 16:9 camera frame into whatever aspect the checkpoint
                # declares (often square). Measured divergence from the
                # geometry training actually saw: ~8.27 deg vs. ~3.2-4.1 deg
                # for the aspect-preserving "pad" default -- see the README.
                arr = np.asarray(
                    Image.fromarray(arr).resize((target_w, target_h), Image.BILINEAR),
                    dtype=np.uint8,
                )
            elif self._fit == "pad":
                arr = self._resize_with_pad(arr, target_h, target_w)
            else:
                # Unreachable via config (as_choice over IMAGE_FITS gates
                # it), but spelled out rather than left as an `else: pad`
                # fallthrough: a third mode added to IMAGE_FITS later would
                # otherwise silently inherit padding instead of failing
                # until someone noticed the geometry was wrong.
                raise ObservationError(f"unrecognized image_fit {self._fit!r}")
        try:
            return encode_image(arr, encoding=self._encoding, quality=self._quality)
        except WireError as exc:
            # Wrap wire.py's own exception type into this module's, so a
            # caller catching ObservationError doesn't also have to know
            # about WireError to cover every rejection path this method can
            # raise -- the same convention gripper.py uses for config_util.
            raise ObservationError(f"could not encode image for {key!r}: {exc}") from exc

    @staticmethod
    def _resize_with_pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize preserving aspect ratio, padding black on the LEFT and TOP.

        Mirrors lerobot's `resize_with_pad`
        (lerobot/policies/common/vla_utils.py:219) -- smolvla's own
        convention, computed by scaling by `ratio = max(cur_w/target_w,
        cur_h/target_h)` so the resized image never exceeds the target in
        either dimension, then padding whatever's left on the left/top with
        zeros (black). This is NOT the centered-padding variant lerobot
        also ships (:142, openpi's convention) -- padding the wrong side
        would be as wrong as stretching, since the policy never saw that
        geometry during training either. Handles upscaling (source smaller
        than target) the same way: `ratio` just comes out < 1.
        """
        cur_h, cur_w = arr.shape[:2]
        ratio = max(cur_w / target_w, cur_h / target_h)
        resized_h = max(1, int(cur_h / ratio))
        resized_w = max(1, int(cur_w / ratio))
        resized = np.asarray(
            Image.fromarray(arr).resize((resized_w, resized_h), Image.BILINEAR), dtype=np.uint8
        )

        pad_h = max(0, target_h - resized_h)
        pad_w = max(0, target_w - resized_w)
        padded_shape = (target_h, target_w) + arr.shape[2:]
        padded = np.zeros(padded_shape, dtype=np.uint8)
        padded[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
        return padded

    def _build_state(self, joint_degrees: list[float], gripper_value: float | None) -> np.ndarray:
        selected = []
        for idx in self._indices:
            if idx < 0 or idx >= len(joint_degrees):
                raise ObservationError(
                    f"state_joint_indices references joint index {idx}, but the arm reports "
                    f"{len(joint_degrees)} joints"
                )
            selected.append(joint_degrees[idx])

        converted = from_degrees(np.asarray(selected, dtype=np.float32), self._units)

        if not self._gripper.in_state:
            return converted

        if not self._gripper.has_normalized_tail:
            idx = self._gripper.arm_joint_index
            if idx < 0 or idx >= len(joint_degrees):
                raise ObservationError(
                    f"gripper joint index {idx} exceeds the arm's {len(joint_degrees)} joints"
                )
            tail = from_degrees(
                np.asarray([joint_degrees[idx]], dtype=np.float32), self._units
            )
        else:
            tail = np.asarray([gripper_value], dtype=np.float32)  # already normalized

        return np.concatenate([converted, tail])
