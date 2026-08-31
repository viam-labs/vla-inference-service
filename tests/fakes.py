"""Minimal fakes standing in for Viam resources.

These are duck-typed stand-ins, not subclasses of the real SDK component
classes -- Tasks 12, 14, and 17 all build on them. Faithfulness matters more
than convenience here: a fake that is too permissive lets a real bug (wrong
units, wrong joint count, an unimplemented SDK method) pass a test that
should have caught it.
"""

from __future__ import annotations

import numpy as np


def default_pose(x=305.4, y=-12.75, z=231.9, o_x=0.0139, o_y=-0.0271, o_z=-0.9995, theta=41.7):
    """A near-vertical tool pose in the shape an xarm reports it.

    Not the identity, and deliberately so. `o_z` near -1 with a non-zero
    `theta` is the regime the recorded dataset actually sits in, and it is the
    regime where a transposed rotation matrix is *closest* to being mistaken
    for the real one -- the rows and columns of this pose's matrix differ by
    0.047, small enough to look like noise and far too large to be one.
    """
    from viam.proto.common import Pose

    return Pose(x=x, y=y, z=z, o_x=o_x, o_y=o_y, o_z=o_z, theta=theta)


class FakeArm:
    """Duck-types `viam.components.arm.Arm` as it exists in the INSTALLED SDK.

    The installed viam-sdk 0.80.0 -- the latest on PyPI -- exposes only
    `get_joint_positions`, `move_to_joint_positions`, `get_end_position`,
    `move_to_position`, `stop`, and `get_kinematics`. It has NO
    `move_through_joint_positions`, and `MoveOptions` is a generated proto type
    that no method consumes; both exist only in the unreleased dev checkout.
    This fake deliberately omits them, so a caller that reaches for the newer
    API fails here rather than on a robot.

    The pose half backs `action_space="delta-ee"`. `move_to_position` snaps the
    reported pose to whatever was commanded, mirroring how
    `move_to_joint_positions` snaps `positions` -- so a test that ticks twice
    sees the second delta composed onto the first result, which is the property
    the relative action space depends on. `PoselessArm` and `RefusingArm` below
    cover the two ways a real driver declines.
    """

    def __init__(self, positions=None, pose=None):
        self.positions = list(positions or [0.0] * 6)
        self.moves = []
        self.move_extras = []
        self.stopped = 0
        self.fail_next_move = False
        self.pose = pose if pose is not None else default_pose()
        self.pose_moves = []
        self.fail_next_pose_move = False

    async def get_end_position(self, **kwargs):
        return self.pose

    async def move_to_position(self, pose, *, extra=None, timeout=None, **kwargs):
        if self.fail_next_pose_move:
            self.fail_next_pose_move = False
            raise RuntimeError("arm could not plan to the requested pose")
        self.pose_moves.append(pose)
        self.pose = pose

    async def get_joint_positions(self, **kwargs):
        from viam.proto.component.arm import JointPositions

        return JointPositions(values=self.positions)

    def _record_move(self, positions, extra):
        """The bookkeeping both arm fakes share.

        Split out so `StalledArm` can override only the part it means to
        change. It previously duplicated this whole prologue to skip one
        trailing line, which is how it came to need its own guard test -- the
        `move_extras` append had to be remembered in two places.
        """
        if self.fail_next_move:
            raise RuntimeError("arm move failed")
        self.moves.append(positions)
        self.move_extras.append(extra)

    async def move_to_joint_positions(self, positions, *, extra=None, timeout=None, **kwargs):
        self._record_move(positions, extra)
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
        self._record_move(positions, extra)
        # Deliberately does NOT update self.positions -- the whole point, and
        # now the only line this override exists to omit.


class PoselessArm(FakeArm):
    """An arm whose driver does not implement `get_end_position`.

    Real drivers signal this by raising, not by returning `None`, so this
    raises. The controller must refuse at startup rather than on the first
    tick.
    """

    async def get_end_position(self, **kwargs):
        raise NotImplementedError("this arm does not report an end position")


class RefusingArm(FakeArm):
    """An arm whose IK refuses every `move_to_position`.

    The kinematically-unreachable-target case: the driver declines *before*
    commanding motion, so the arm is stationary and the reported pose never
    changes. `refusals` counts the attempts, which is what proves the loop
    kept ticking rather than halting on the first one.
    """

    def __init__(self, positions=None, pose=None):
        super().__init__(positions=positions, pose=pose)
        self.refusals = 0

    async def move_to_position(self, pose, *, extra=None, timeout=None, **kwargs):
        self.refusals += 1
        raise RuntimeError("cannot plan to the requested pose: target unreachable")



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


class FakeDoCommandGripper:
    """A gripper whose only proportional control is through ``DoCommand``.

    Mirrors the contract both `devrel:so101:gripper` and
    `viam:ufactory:gripper` implement: ``{"get": True}`` returns the current
    position under some key, ``{"set": n}`` commands a new one.

    `position` is untyped and simply echoed back, so a non-numeric or `None`
    value (a driver returning a bad payload, or JSON null) reaches the
    caller unchanged rather than needing a dedicated knob for it. Likewise
    a driver answering under an unexpected key is just `read_key` set to
    something other than what the caller configured. An unrecognized
    command raises `AssertionError` -- kept rather than a bare `assert` so
    it survives `-O` -- but note that inside the controller tick loop
    (`src/vla/controller/service.py`), broad `except Exception` handlers
    launder this into `last_error` text rather than letting it propagate as
    a stack trace; a service-level test must assert on `last_error` to see
    it.
    """

    def __init__(self, position=0.0, read_key="position"):
        self.position = position
        self.read_key = read_key
        self.commands = []

    async def do_command(self, command, *, timeout=None, **kwargs):
        self.commands.append(dict(command))
        if command.get("get") is True:
            return {self.read_key: self.position}
        if "set" in command:
            self.position = command["set"]
            return {self.read_key: self.position}
        raise AssertionError(f"unexpected command {command!r}")
