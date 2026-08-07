"""Tests for concurrent observation assembly.

Camera reads use the installed viam-sdk's real API: `get_images()` (plural),
returning `(Sequence[NamedImage], ResponseMetadata)` -- not the `get_image()`
singular that an earlier draft of this plan assumed and does not exist on
0.80.0. `tests/fakes.py::FakeCamera` was fixed to match before this file was
written; a fake matching a nonexistent method would let every test below
pass and still fail against a real robot.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from vla.controller.gripper import make_gripper_adapter
from vla.controller.observation import (
    DEFAULT_DURATION_WARN_S,
    STALE_FRAME_WARN_S,
    ObservationBuilder,
    ObservationError,
)
from vla.wire import decode_image
from tests.fakes import FakeArm, FakeCamera, FakeServo


def _builder(**kw):
    defaults = dict(
        cameras={"observation.images.top": FakeCamera()},
        arm=FakeArm(positions=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        gripper=make_gripper_adapter({"type": "none"}, {}),
        state_joint_indices=[0, 1, 2, 3, 4],
        state_units="degrees",
        image_sizes={"observation.images.top": (224, 224)},
        image_encoding="jpeg",
        jpeg_quality=90,
    )
    defaults.update(kw)
    return ObservationBuilder(**defaults)


# ---------------------------------------------------------------------------
# state assembly
# ---------------------------------------------------------------------------


async def test_builds_state_from_selected_indices():
    obs = await _builder().build()
    np.testing.assert_allclose(obs.state, [10.0, 20.0, 30.0, 40.0, 50.0])


async def test_state_converted_to_radians():
    obs = await _builder(state_units="radians").build()
    np.testing.assert_allclose(obs.state, np.deg2rad([10.0, 20.0, 30.0, 40.0, 50.0]), rtol=1e-5)


async def test_state_stays_in_degrees_by_default():
    # Assert the default explicitly -- a suite that only checks the radians
    # override would stay green if the default silently changed unit.
    obs = await _builder().build()
    np.testing.assert_allclose(obs.state, [10.0, 20.0, 30.0, 40.0, 50.0])


async def test_joint_reordering_is_respected():
    obs = await _builder(state_joint_indices=[4, 3, 2, 1, 0]).build()
    np.testing.assert_allclose(obs.state, [50.0, 40.0, 30.0, 20.0, 10.0])


async def test_arm_joint_gripper_appended_from_arm():
    obs = await _builder(
        gripper=make_gripper_adapter({"type": "arm_joint", "joint_index": 5}, {})
    ).build()
    np.testing.assert_allclose(obs.state, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])


async def test_servo_gripper_appended_normalized():
    servo = FakeServo(angle=45)
    gripper = make_gripper_adapter(
        {"type": "servo", "name": "s", "min_deg": 0, "max_deg": 90}, {"s": servo}
    )
    obs = await _builder(gripper=gripper).build()
    assert obs.state[-1] == pytest.approx(0.5)


async def test_no_gripper_leaves_state_unmodified():
    # Assert the "none" default doesn't silently append anything.
    obs = await _builder().build()
    assert obs.state.shape == (5,)


async def test_out_of_range_joint_index_errors():
    with pytest.raises(ObservationError, match="joint index"):
        await _builder(state_joint_indices=[0, 99]).build()


async def test_negative_joint_index_errors():
    with pytest.raises(ObservationError, match="joint index"):
        await _builder(state_joint_indices=[0, -1]).build()


async def test_arm_joint_gripper_index_out_of_range_errors():
    with pytest.raises(ObservationError, match="gripper joint index"):
        await _builder(
            gripper=make_gripper_adapter({"type": "arm_joint", "joint_index": 99}, {})
        ).build()


# ---------------------------------------------------------------------------
# image assembly
# ---------------------------------------------------------------------------


async def test_images_resized_to_policy_expectation():
    obs = await _builder().build()
    img = decode_image(obs.images["observation.images.top"])
    assert img.shape == (224, 224, 3)


async def test_image_encoding_field_matches_configured_encoding():
    obs = await _builder(image_encoding="png").build()
    assert obs.images["observation.images.top"]["encoding"] == "png"


async def test_missing_expected_size_raises_before_encoding():
    # The builder must never forward an unresized frame -- both because a
    # 1080p raw frame is ~8.3 MB base64-encoded (past typical gRPC limits)
    # and because the policy would receive a resolution it wasn't trained on.
    with pytest.raises(ObservationError, match="observation.images.top"):
        await _builder(image_sizes={}).build()


async def test_unknown_image_encoding_raises_observation_error_not_wire_error():
    with pytest.raises(ObservationError, match="encoding"):
        await _builder(image_encoding="webp").build()


# ---------------------------------------------------------------------------
# camera failure / emptiness
# ---------------------------------------------------------------------------


async def test_camera_failure_fails_the_whole_tick():
    # Never substitute a black frame or reuse a stale one: both silently
    # corrupt policy input in ways that look like bad model behavior.
    with pytest.raises(ObservationError, match="camera"):
        await _builder(cameras={"observation.images.top": FakeCamera(fail=True)}).build()


async def test_camera_returning_zero_images_raises_observation_error_not_index_error():
    with pytest.raises(ObservationError, match="observation.images.top") as exc_info:
        await _builder(cameras={"observation.images.top": FakeCamera(empty=True)}).build()
    assert not isinstance(exc_info.value, IndexError)
    # The empty-images error is already specific; it must surface as-is,
    # not get double-wrapped into "camera read failed: camera returned zero
    # images" by the generic exception-from-gather handling.
    assert str(exc_info.value) == "camera 'observation.images.top' returned zero images"


async def test_one_failing_camera_among_several_still_fails_the_tick():
    cams = {
        "observation.images.top": FakeCamera(),
        "observation.images.wrist": FakeCamera(fail=True),
    }
    sizes = {k: (224, 224) for k in cams}
    with pytest.raises(ObservationError, match="observation.images.wrist"):
        await _builder(cameras=cams, image_sizes=sizes).build()


# ---------------------------------------------------------------------------
# concurrency and timing
# ---------------------------------------------------------------------------


async def test_cameras_are_read_concurrently():
    # Serial reads would take ~3x one camera's latency. At 10 Hz the whole
    # tick budget is 100 ms, so this is a real constraint, not a nicety.
    class SlowCamera(FakeCamera):
        async def get_images(self, *a, **k):
            await asyncio.sleep(0.05)
            return await super().get_images(*a, **k)

    cams = {f"observation.images.c{i}": SlowCamera() for i in range(3)}
    b = _builder(cameras=cams, image_sizes={k: (224, 224) for k in cams})

    obs = await b.build()

    assert len(obs.images) == 3
    assert all(c.reads == 1 for c in cams.values())
    assert obs.duration_s < 0.12, f"reads look serial: {obs.duration_s:.3f}s for 3x50ms"


async def test_duration_reflects_actual_assembly_time():
    class SlowCamera(FakeCamera):
        async def get_images(self, *a, **k):
            await asyncio.sleep(0.05)
            return await super().get_images(*a, **k)

    obs = await _builder(cameras={"observation.images.top": SlowCamera()}).build()
    assert obs.duration_s >= 0.05


async def test_slow_assembly_logs_a_duration_warning(caplog):
    class SlowCamera(FakeCamera):
        async def get_images(self, *a, **k):
            await asyncio.sleep(DEFAULT_DURATION_WARN_S + 0.05)
            return await super().get_images(*a, **k)

    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await _builder(cameras={"observation.images.top": SlowCamera()}).build()

    assert any("assembly" in r.message and "budget" in r.message for r in caplog.records)


async def test_fast_assembly_does_not_log_a_duration_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await _builder().build()

    assert not any("budget" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# captured_at staleness -- distinct from the duration warning above: this
# catches a camera silently serving a buffered old frame, which even an
# instantaneous read cannot detect via duration alone.
# ---------------------------------------------------------------------------


async def test_stale_captured_at_logs_a_staleness_warning(caplog):
    stale_at = datetime.now(timezone.utc) - timedelta(seconds=STALE_FRAME_WARN_S + 1.0)
    cam = FakeCamera(captured_at=stale_at)

    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await _builder(cameras={"observation.images.top": cam}).build()

    assert any(
        "observation.images.top" in r.message and "stale" in r.message for r in caplog.records
    )


async def test_fresh_captured_at_does_not_log_a_staleness_warning(caplog):
    fresh_at = datetime.now(timezone.utc)
    cam = FakeCamera(captured_at=fresh_at)

    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await _builder(cameras={"observation.images.top": cam}).build()

    assert not any("stale" in r.message for r in caplog.records)


async def test_unset_captured_at_does_not_log_a_staleness_warning(caplog):
    # A driver that never populates ResponseMetadata leaves captured_at at
    # its zero-value default; that must not be misread as "1970" staleness.
    cam = FakeCamera(captured_at=None, populate_metadata=False)

    with caplog.at_level(logging.WARNING, logger="vla.controller.observation"):
        await _builder(cameras={"observation.images.top": cam}).build()

    assert not any("stale" in r.message for r in caplog.records)
