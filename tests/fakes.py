"""Minimal fakes standing in for Viam resources.

These are duck-typed stand-ins, not subclasses of the real SDK component
classes -- Tasks 12, 14, and 17 all build on them. Faithfulness matters more
than convenience here: a fake that is too permissive lets a real bug (wrong
units, wrong joint count, an unimplemented SDK method) pass a test that
should have caught it.
"""

from __future__ import annotations

import numpy as np


class FakeArm:
    """Duck-types `viam.components.arm.Arm` as it exists in the INSTALLED SDK.

    The installed viam-sdk 0.80.0 -- the latest on PyPI -- exposes only
    `get_joint_positions`, `move_to_joint_positions`, `stop`, and
    `get_kinematics`. It has NO `move_through_joint_positions`, and `MoveOptions`
    is a generated proto type that no method consumes; both exist only in the
    unreleased dev checkout. This fake deliberately omits them, so a caller that
    reaches for the newer API fails here rather than on a robot.
    """

    def __init__(self, positions=None):
        self.positions = list(positions or [0.0] * 6)
        self.moves = []
        self.stopped = 0
        self.fail_next_move = False

    async def get_joint_positions(self, **kwargs):
        from viam.proto.component.arm import JointPositions

        return JointPositions(values=self.positions)

    async def move_to_joint_positions(self, positions, *, extra=None, timeout=None, **kwargs):
        if self.fail_next_move:
            raise RuntimeError("arm move failed")
        self.moves.append(positions)
        # Write into the existing vector rather than replacing it: a commanded
        # action can be narrower than the arm's joint count (gripper on its own
        # component), and replacing would silently shrink the arm.
        commanded = list(positions.values)
        self.positions[: len(commanded)] = commanded

    async def stop(self, **kwargs):
        self.stopped += 1


class StalledArm(FakeArm):
    """An arm that accepts move commands but whose *measured* position never
    changes -- simulating a jammed/stalled joint.

    `FakeArm` snaps `self.positions` to whatever was last commanded, so with
    it, "the measured position" and "the last commanded position" are always
    identical and indistinguishable to a test. The safety layer's delta clamp
    is specifically supposed to clamp against the *measured* position on
    every tick, not the last commanded one (see `safety.py`'s docstring:
    "so a stalled arm cannot accumulate an ever-growing command") -- a
    controller-level regression that swapped one for the other (e.g. caching
    `current` outside the loop, or feeding the previous `safe` back in as the
    next tick's `current`) would pass every test built on `FakeArm` alone.
    This fake exists so that specific property has real coverage.
    """

    async def move_to_joint_positions(self, positions, *, extra=None, timeout=None, **kwargs):
        if self.fail_next_move:
            raise RuntimeError("arm move failed")
        self.moves.append(positions)
        # Deliberately does NOT update self.positions -- the whole point.


class FakeCamera:
    """Duck-types `viam.components.camera.Camera`.

    The real API is `get_images()` (plural), returning
    `(Sequence[NamedImage], ResponseMetadata)` -- not `get_image()`, which
    does not exist on installed viam-sdk 0.80.0. A fake implementing the
    nonexistent singular method would let observation-assembly tests pass
    while failing against a real robot, so this fake matches the real
    signature exactly.
    """

    def __init__(self, size=(480, 640), fail=False, empty=False, captured_at=None, populate_metadata=True):
        self.size = size
        self.fail = fail
        self.empty = empty
        # `None` (the default) means "use the current time" -- a fresh
        # frame. Pass an explicit `datetime` to simulate a camera serving a
        # buffered, stale frame. `populate_metadata=False` simulates a
        # driver that never sets `captured_at` at all, leaving it at protobuf's
        # zero-value default -- distinct from "captured just now".
        self.captured_at = captured_at
        self.populate_metadata = populate_metadata
        self.reads = 0

    async def get_images(self, *args, **kwargs):
        from viam.media.video import NamedImage, CameraMimeType
        from viam.proto.common import ResponseMetadata
        from google.protobuf.timestamp_pb2 import Timestamp
        import io
        from PIL import Image

        self.reads += 1
        if self.fail:
            raise RuntimeError("camera read failed")

        metadata = ResponseMetadata()
        if self.populate_metadata:
            ts = Timestamp()
            if self.captured_at is not None:
                ts.FromDatetime(self.captured_at)
            else:
                ts.GetCurrentTime()
            metadata.captured_at.CopyFrom(ts)

        if self.empty:
            return [], metadata

        h, w = self.size
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        image = NamedImage("image", buf.getvalue(), CameraMimeType.JPEG)
        return [image], metadata


class FakeServo:
    def __init__(self, angle=0):
        self.angle = angle
        self.moves = []

    async def get_position(self, **kwargs) -> int:
        return self.angle

    async def move(self, angle: int, **kwargs):
        self.moves.append(angle)
        self.angle = angle


class FakeGripper:
    def __init__(self, inputs=None, supports_inputs=True):
        self.inputs = list(inputs or [0.0])
        self.supports_inputs = supports_inputs
        self.opened = 0
        self.grabbed = 0
        self.sent = []

    async def get_current_inputs(self, **kwargs):
        return list(self.inputs)

    async def go_to_inputs(self, values, **kwargs):
        if not self.supports_inputs:
            raise NotImplementedError("go_to_inputs unimplemented")
        self.sent.append(list(values))
        self.inputs = list(values)

    async def open(self, **kwargs):
        self.opened += 1

    async def grab(self, **kwargs):
        self.grabbed += 1
        return True
