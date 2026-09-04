"""Tests for viam-labs:vla:controller config parsing and validation.

Uses `vla.config_util`'s `ConfigError`/`as_int`/`as_float`/`as_choice`
throughout -- there is exactly one `ConfigError` type in this project, so a
caller only ever needs `except ConfigError` to cover every rejection path
here (and in `vla.policy.config`).

`MoveOptions`-derived fields (`max_acc_degs_per_sec2`,
`max_tcp_speed_m_per_sec`) do not exist in this config: `move_through_joint_
positions`/`MoveOptions` ship in no released viam-sdk (installed 0.80.0 has
only `move_to_joint_positions`), so those fields had no enforcement path and
were dropped rather than kept as knobs that silently do nothing.
`max_vel_degs_per_sec` is the one that survives, because the safety layer's
existing `max_joint_delta_degs` per-tick clamp is directly derivable from it
given `fps`.
"""

import pytest

from vla.controller.config import (
    DEFAULT_ARM_MOVE_EXTRA,
    ConfigError,
    ControllerConfig,
    SafetyConfig,
)
from vla.controller.gripper import GRIPPER_TYPES, make_gripper_adapter


BASE = {
    "policy_service": "vla-policy",
    "arm": "my-arm",
    "cameras": {"observation.images.top": "cam-top"},
    "state_joint_indices": [0, 1, 2, 3, 4],
}


# ---------------------------------------------------------------------------
# Minimal parse + defaults (assert defaults explicitly, not just overrides).
# ---------------------------------------------------------------------------


def test_parses_minimal_config():
    cfg = ControllerConfig.parse(BASE)
    assert cfg.policy_service == "vla-policy"
    assert cfg.arm == "my-arm"
    assert cfg.cameras == {"observation.images.top": "cam-top"}
    assert cfg.state_joint_indices == [0, 1, 2, 3, 4]
    assert cfg.gripper == {"type": "none"}
    assert cfg.task == ""
    assert cfg.fps == 10.0
    assert cfg.mode == "auto"
    # None, not a fixed number -- the right value is derived from the
    # checkpoint's n_action_steps once specs are known (see config.py's
    # docstring); config parsing alone has no access to that.
    assert cfg.queue_threshold is None
    assert cfg.starvation_grace_ticks == 3
    assert cfg.policy_ready_timeout_s == 600
    assert cfg.state_units == "degrees"
    assert cfg.action_units == "degrees"
    assert cfg.image_encoding == "jpeg"
    assert cfg.jpeg_quality == 90
    assert cfg.image_fit == "pad"


def test_default_safety_config():
    cfg = ControllerConfig.parse(BASE)
    assert cfg.safety.max_joint_delta_degs == 8.0
    assert cfg.safety.max_start_delta_degs == 15.0
    assert cfg.safety.max_vel_degs_per_sec is None
    assert cfg.safety.joint_limits_degs is None
    assert cfg.safety.stop_on_error is True


def test_removed_move_options_fields_do_not_exist():
    # These knobs have no enforcement path on any released viam-sdk
    # (move_through_joint_positions / MoveOptions ship in none) -- a silent
    # no-op knob is worse than an absent one, so they were deleted rather
    # than kept unused.
    cfg = ControllerConfig.parse(BASE)
    assert not hasattr(cfg.safety, "max_acc_degs_per_sec2")
    assert not hasattr(cfg.safety, "max_tcp_speed_m_per_sec")


# ---------------------------------------------------------------------------
# dependencies()
# ---------------------------------------------------------------------------


def test_dependencies_include_policy_arm_and_cameras():
    assert set(ControllerConfig.parse(BASE).dependencies()) == {
        "vla-policy",
        "my-arm",
        "cam-top",
    }


def test_servo_gripper_adds_dependency():
    cfg = ControllerConfig.parse(
        {**BASE, "gripper": {"type": "servo", "name": "grip", "min_deg": 0, "max_deg": 90}}
    )
    assert "grip" in cfg.dependencies()


def test_do_command_gripper_adds_dependency():
    cfg = ControllerConfig.parse(
        {**BASE, "gripper": {"type": "do_command", "name": "g2", "open_value": 95, "closed_value": 0}}
    )
    assert "g2" in cfg.dependencies()


def test_arm_joint_gripper_adds_no_dependency():
    cfg = ControllerConfig.parse({**BASE, "gripper": {"type": "arm_joint", "joint_index": 5}})
    assert set(cfg.dependencies()) == {"vla-policy", "my-arm", "cam-top"}


def test_no_gripper_adds_no_dependency():
    assert set(ControllerConfig.parse(BASE).dependencies()) == {
        "vla-policy",
        "my-arm",
        "cam-top",
    }


# ---------------------------------------------------------------------------
# required fields
# ---------------------------------------------------------------------------


def test_requires_policy_service():
    with pytest.raises(ConfigError, match="policy_service"):
        ControllerConfig.parse({k: v for k, v in BASE.items() if k != "policy_service"})


def test_requires_arm():
    with pytest.raises(ConfigError, match="arm"):
        ControllerConfig.parse({k: v for k, v in BASE.items() if k != "arm"})


def test_requires_at_least_one_camera():
    with pytest.raises(ConfigError, match="cameras"):
        ControllerConfig.parse({**BASE, "cameras": {}})


def test_requires_state_joint_indices():
    with pytest.raises(ConfigError, match="state_joint_indices"):
        ControllerConfig.parse({k: v for k, v in BASE.items() if k != "state_joint_indices"})


# ---------------------------------------------------------------------------
# fps
# ---------------------------------------------------------------------------


def test_rejects_nonpositive_fps():
    with pytest.raises(ConfigError, match="fps"):
        ControllerConfig.parse({**BASE, "fps": 0})


def test_rejects_negative_fps():
    with pytest.raises(ConfigError, match="fps"):
        ControllerConfig.parse({**BASE, "fps": -10.0})


def test_accepts_fractional_fps():
    cfg = ControllerConfig.parse({**BASE, "fps": 7.5})
    assert cfg.fps == 7.5


def test_fps_as_protobuf_double_is_accepted():
    # Struct delivers every number as a double; 10 arrives as 10.0.
    cfg = ControllerConfig.parse({**BASE, "fps": 10.0})
    assert cfg.fps == 10.0


# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------


def test_rejects_unknown_mode():
    with pytest.raises(ConfigError, match="mode"):
        ControllerConfig.parse({**BASE, "mode": "turbo"})


@pytest.mark.parametrize("mode", ["auto", "sequential", "rtc", "async"])
def test_accepts_every_known_mode(mode):
    # Hardcoded literal, not MODES itself -- a mutant that shrinks MODES
    # must not be able to shrink this test's coverage along with it.
    cfg = ControllerConfig.parse({**BASE, "mode": mode})
    assert cfg.mode == mode


# ---------------------------------------------------------------------------
# state_joint_indices
# ---------------------------------------------------------------------------


def test_rejects_duplicate_joint_indices():
    with pytest.raises(ConfigError, match="duplicate"):
        ControllerConfig.parse({**BASE, "state_joint_indices": [0, 1, 1]})


def test_rejects_empty_joint_indices():
    with pytest.raises(ConfigError, match="state_joint_indices"):
        ControllerConfig.parse({**BASE, "state_joint_indices": []})


def test_state_joint_indices_reordering_is_preserved():
    cfg = ControllerConfig.parse({**BASE, "state_joint_indices": [4, 3, 2, 1, 0]})
    assert cfg.state_joint_indices == [4, 3, 2, 1, 0]


def test_state_joint_indices_accepts_protobuf_double_form():
    cfg = ControllerConfig.parse({**BASE, "state_joint_indices": [0.0, 1.0, 2.0]})
    assert cfg.state_joint_indices == [0, 1, 2]


def test_state_joint_indices_rejects_fractional_value():
    with pytest.raises(ConfigError, match="state_joint_indices"):
        ControllerConfig.parse({**BASE, "state_joint_indices": [0, 1.5]})


# ---------------------------------------------------------------------------
# gripper block
# ---------------------------------------------------------------------------


def test_rejects_unknown_gripper_type():
    with pytest.raises(ConfigError, match="gripper"):
        ControllerConfig.parse({**BASE, "gripper": {"type": "claw"}})


# Keyed by kind, not a positional/chained-conditional shape, so that adding a
# new GRIPPER_TYPES entry without a matching key here fails loudly (KeyError)
# in every test below that indexes it, instead of silently falling through a
# stale branch the way an `if kind == ...` chain would.
GRIPPER_EXTRA = {
    "arm_joint": {"joint_index": 5},
    "servo": {"name": "grip"},
    "gripper": {"name": "grip"},
    "do_command": {"name": "grip", "open_value": 95.0, "closed_value": 0.0},
    "none": {},
}


@pytest.mark.parametrize("kind", GRIPPER_TYPES)
def test_accepts_every_known_gripper_type(kind):
    cfg = ControllerConfig.parse({**BASE, "gripper": {"type": kind, **GRIPPER_EXTRA[kind]}})
    assert cfg.gripper["type"] == kind


@pytest.mark.parametrize("kind", GRIPPER_TYPES)
def test_dependencies_cover_what_the_adapter_needs(kind):
    """`dependencies()` must name every resource the built adapter will ask for.

    It has to guess before any adapter exists, from a hardcoded tuple of type
    strings; the adapter's own `dependency_name` is the truth, available only
    after construction. Deriving the parametrization from GRIPPER_TYPES means a
    new variant whose type was never added to that tuple fails here, at CI,
    instead of as an AttributeError at robot start.
    """
    cfg = ControllerConfig.parse({**BASE, "gripper": {"type": kind, **GRIPPER_EXTRA[kind]}})
    adapter = make_gripper_adapter(cfg.gripper, {"grip": object()})
    if adapter.dependency_name:
        assert adapter.dependency_name in cfg.dependencies()


# ---------------------------------------------------------------------------
# joint_limits_degs (safety block) -- indexed in action-vector order, with a
# trailing gripper pair only when gripper.type == "arm_joint".
# ---------------------------------------------------------------------------


def test_joint_limits_length_must_match_arm_joint_gripper():
    with pytest.raises(ConfigError, match="joint_limits_degs"):
        ControllerConfig.parse(
            {
                **BASE,
                "gripper": {"type": "arm_joint", "joint_index": 5},
                "safety": {"joint_limits_degs": [[-90, 90]] * 5},  # needs 6
            }
        )


def test_joint_limits_length_matches_with_arm_joint_gripper():
    cfg = ControllerConfig.parse(
        {
            **BASE,
            "gripper": {"type": "arm_joint", "joint_index": 5},
            "safety": {"joint_limits_degs": [[-90, 90]] * 6},
        }
    )
    assert len(cfg.safety.joint_limits_degs) == 6


def test_joint_limits_length_matches_without_degree_gripper():
    cfg = ControllerConfig.parse(
        {
            **BASE,
            "gripper": {"type": "servo", "name": "grip"},
            "safety": {"joint_limits_degs": [[-90, 90]] * 5},  # no trailing pair
        }
    )
    assert len(cfg.safety.joint_limits_degs) == 5


def test_joint_limits_length_too_short_by_one_is_rejected():
    with pytest.raises(ConfigError, match="joint_limits_degs"):
        ControllerConfig.parse({**BASE, "safety": {"joint_limits_degs": [[-90, 90]] * 4}})


def test_joint_limits_length_too_long_by_one_is_rejected():
    with pytest.raises(ConfigError, match="joint_limits_degs"):
        ControllerConfig.parse({**BASE, "safety": {"joint_limits_degs": [[-90, 90]] * 6}})


def test_rejects_inverted_joint_limit():
    with pytest.raises(ConfigError, match="min"):
        ControllerConfig.parse({**BASE, "safety": {"joint_limits_degs": [[90, -90]] * 5}})


def test_rejects_equal_joint_limit_bounds():
    with pytest.raises(ConfigError, match="min"):
        ControllerConfig.parse({**BASE, "safety": {"joint_limits_degs": [[0, 0]] * 5}})


def test_no_joint_limits_by_default():
    cfg = ControllerConfig.parse(BASE)
    assert cfg.safety.joint_limits_degs is None


# ---------------------------------------------------------------------------
# max_vel_degs_per_sec / max_joint_delta_degs -- see the Task 17 BLOCKER
# RESOLVED callout: MoveOptions velocity ceilings have no enforcement path
# on any released SDK, so the velocity bound lives entirely in the safety
# layer's existing per-tick delta clamp. max_vel_degs_per_sec is the
# operator-facing knob (reasoning in deg/s is what a human can actually do);
# max_joint_delta_degs is derived from it (= max_vel_degs_per_sec / fps) and
# is read-only when derived. Both may appear in config, but if both are
# given they must agree, or it's a config-time error.
# ---------------------------------------------------------------------------


def test_max_vel_derives_max_joint_delta():
    cfg = ControllerConfig.parse({**BASE, "fps": 20.0, "safety": {"max_vel_degs_per_sec": 100.0}})
    assert cfg.safety.max_vel_degs_per_sec == 100.0
    assert cfg.safety.max_joint_delta_degs == pytest.approx(5.0)  # 100 / 20


def test_max_joint_delta_alone_is_still_accepted():
    # Legacy / direct path: no velocity knob given at all.
    cfg = ControllerConfig.parse({**BASE, "safety": {"max_joint_delta_degs": 3.0}})
    assert cfg.safety.max_joint_delta_degs == 3.0
    assert cfg.safety.max_vel_degs_per_sec is None


def test_consistent_max_vel_and_max_joint_delta_are_both_accepted():
    cfg = ControllerConfig.parse(
        {
            **BASE,
            "fps": 10.0,
            "safety": {"max_vel_degs_per_sec": 50.0, "max_joint_delta_degs": 5.0},
        }
    )
    assert cfg.safety.max_joint_delta_degs == 5.0
    assert cfg.safety.max_vel_degs_per_sec == 50.0


def test_inconsistent_max_vel_and_max_joint_delta_is_rejected():
    with pytest.raises(ConfigError, match="max_vel_degs_per_sec"):
        ControllerConfig.parse(
            {
                **BASE,
                "fps": 10.0,
                # max_vel/fps = 5.0, but max_joint_delta_degs says 8.0.
                "safety": {"max_vel_degs_per_sec": 50.0, "max_joint_delta_degs": 8.0},
            }
        )


def test_default_max_joint_delta_when_neither_vel_nor_delta_given():
    cfg = ControllerConfig.parse({**BASE, "fps": 20.0})
    assert cfg.safety.max_joint_delta_degs == 8.0
    assert cfg.safety.max_vel_degs_per_sec is None


def test_rejects_nonpositive_max_vel_degs_per_sec():
    with pytest.raises(ConfigError, match="max_vel_degs_per_sec"):
        ControllerConfig.parse({**BASE, "safety": {"max_vel_degs_per_sec": 0}})


def test_rejects_negative_max_joint_delta_degs():
    with pytest.raises(ConfigError, match="max_joint_delta_degs"):
        ControllerConfig.parse({**BASE, "safety": {"max_joint_delta_degs": -1.0}})


def test_max_vel_degs_per_sec_derivation_uses_fractional_fps():
    cfg = ControllerConfig.parse(
        {**BASE, "fps": 7.5, "safety": {"max_vel_degs_per_sec": 15.0}}
    )
    assert cfg.safety.max_joint_delta_degs == pytest.approx(2.0)  # 15 / 7.5


# ---------------------------------------------------------------------------
# units -- state_units: "normalized" must be rejected at config time, not
# the first control tick. units.py's SUPPORTED_UNITS (degrees, radians) is
# what a config must validate against -- UNITS also includes "normalized",
# which units.py can accept as *configuration* but never actually convert.
# ---------------------------------------------------------------------------


def test_rejects_normalized_state_units_at_config_time():
    with pytest.raises(ConfigError, match="state_units"):
        ControllerConfig.parse({**BASE, "state_units": "normalized"})


def test_rejects_normalized_action_units_at_config_time():
    with pytest.raises(ConfigError, match="action_units"):
        ControllerConfig.parse({**BASE, "action_units": "normalized"})


def test_rejects_unknown_state_units():
    with pytest.raises(ConfigError, match="state_units"):
        ControllerConfig.parse({**BASE, "state_units": "furlongs"})


@pytest.mark.parametrize("unit", ["degrees", "radians"])
def test_accepts_every_supported_state_unit(unit):
    cfg = ControllerConfig.parse({**BASE, "state_units": unit})
    assert cfg.state_units == unit


@pytest.mark.parametrize("unit", ["degrees", "radians"])
def test_accepts_every_supported_action_unit(unit):
    cfg = ControllerConfig.parse({**BASE, "action_units": unit})
    assert cfg.action_units == unit


# ---------------------------------------------------------------------------
# image_encoding / jpeg_quality
# ---------------------------------------------------------------------------


def test_rejects_unknown_image_encoding():
    with pytest.raises(ConfigError, match="image_encoding"):
        ControllerConfig.parse({**BASE, "image_encoding": "webp"})


@pytest.mark.parametrize("encoding", ["jpeg", "png", "raw"])
def test_accepts_every_known_image_encoding(encoding):
    cfg = ControllerConfig.parse({**BASE, "image_encoding": encoding})
    assert cfg.image_encoding == encoding


def test_rejects_out_of_range_jpeg_quality():
    with pytest.raises(ConfigError, match="jpeg_quality"):
        ControllerConfig.parse({**BASE, "jpeg_quality": 101})


def test_rejects_negative_jpeg_quality():
    with pytest.raises(ConfigError, match="jpeg_quality"):
        ControllerConfig.parse({**BASE, "jpeg_quality": -1})


def test_accepts_boundary_jpeg_quality_values():
    assert ControllerConfig.parse({**BASE, "jpeg_quality": 0}).jpeg_quality == 0
    assert ControllerConfig.parse({**BASE, "jpeg_quality": 100}).jpeg_quality == 100


def test_jpeg_quality_as_protobuf_double_is_accepted():
    cfg = ControllerConfig.parse({**BASE, "jpeg_quality": 90.0})
    assert cfg.jpeg_quality == 90


def test_rejects_fractional_jpeg_quality():
    with pytest.raises(ConfigError, match="jpeg_quality"):
        ControllerConfig.parse({**BASE, "jpeg_quality": 90.5})


# ---------------------------------------------------------------------------
# image_fit -- "pad" (aspect-preserving, smolvla's resize_with_pad
# convention) is the default; "stretch" is kept only so an existing
# deployment can reproduce its pre-fix output. See observation.py.
# ---------------------------------------------------------------------------


def test_image_fit_defaults_to_pad():
    cfg = ControllerConfig.parse(BASE)
    assert cfg.image_fit == "pad"


@pytest.mark.parametrize("fit", ["pad", "stretch"])
def test_accepts_every_known_image_fit(fit):
    cfg = ControllerConfig.parse({**BASE, "image_fit": fit})
    assert cfg.image_fit == fit


def test_rejects_unknown_image_fit():
    with pytest.raises(ConfigError, match="image_fit"):
        ControllerConfig.parse({**BASE, "image_fit": "crop"})


# ---------------------------------------------------------------------------
# task, queue_threshold, starvation_grace_ticks, policy_ready_timeout_s
# ---------------------------------------------------------------------------


def test_task_is_passed_through():
    cfg = ControllerConfig.parse({**BASE, "task": "pick up the red block"})
    assert cfg.task == "pick up the red block"


def test_queue_threshold_accepts_override():
    cfg = ControllerConfig.parse({**BASE, "queue_threshold": 5})
    assert cfg.queue_threshold == 5


def test_queue_threshold_explicit_zero_is_not_treated_as_unset():
    # 0 is a legitimate explicit value (fire the refill only once the queue
    # is already empty) -- it must not collapse into the same "derive it"
    # behavior as never mentioning the field at all.
    cfg = ControllerConfig.parse({**BASE, "queue_threshold": 0})
    assert cfg.queue_threshold == 0


def test_queue_threshold_rejects_fractional_double():
    with pytest.raises(ConfigError, match="queue_threshold"):
        ControllerConfig.parse({**BASE, "queue_threshold": 5.5})


def test_starvation_grace_ticks_accepts_override():
    cfg = ControllerConfig.parse({**BASE, "starvation_grace_ticks": 7})
    assert cfg.starvation_grace_ticks == 7


def test_policy_ready_timeout_s_accepts_override():
    cfg = ControllerConfig.parse({**BASE, "policy_ready_timeout_s": 30})
    assert cfg.policy_ready_timeout_s == 30


def test_rejects_nonpositive_policy_ready_timeout_s():
    with pytest.raises(ConfigError, match="policy_ready_timeout_s"):
        ControllerConfig.parse({**BASE, "policy_ready_timeout_s": 0})


def test_stop_on_error_accepts_override():
    cfg = ControllerConfig.parse({**BASE, "safety": {"stop_on_error": False}})
    assert cfg.safety.stop_on_error is False


def test_stop_on_error_rejects_non_bool():
    with pytest.raises(ConfigError, match="stop_on_error"):
        ControllerConfig.parse({**BASE, "safety": {"stop_on_error": "false"}})


# ---------------------------------------------------------------------------
# duration_warn_s / stale_frame_warn_s -- operator-configurable pass-through
# to ObservationBuilder. Defaults must match observation.py's own module
# constants exactly, so a controller that never sets these observes no
# behavior change from before they existed as config fields.
# ---------------------------------------------------------------------------


def test_default_duration_warn_s_matches_observation_module_default():
    from vla.controller.observation import DEFAULT_DURATION_WARN_S

    cfg = ControllerConfig.parse(BASE)
    assert cfg.duration_warn_s == DEFAULT_DURATION_WARN_S


def test_default_stale_frame_warn_s_matches_observation_module_default():
    from vla.controller.observation import STALE_FRAME_WARN_S

    cfg = ControllerConfig.parse(BASE)
    assert cfg.stale_frame_warn_s == STALE_FRAME_WARN_S


def test_duration_warn_s_accepts_override():
    cfg = ControllerConfig.parse({**BASE, "duration_warn_s": 0.4})
    assert cfg.duration_warn_s == 0.4


def test_stale_frame_warn_s_accepts_override():
    cfg = ControllerConfig.parse({**BASE, "stale_frame_warn_s": 2.5})
    assert cfg.stale_frame_warn_s == 2.5


def test_duration_warn_s_accepts_protobuf_double_form():
    cfg = ControllerConfig.parse({**BASE, "duration_warn_s": 1.0})
    assert cfg.duration_warn_s == 1.0


def test_duration_warn_s_rejects_negative():
    with pytest.raises(ConfigError, match="duration_warn_s"):
        ControllerConfig.parse({**BASE, "duration_warn_s": -0.1})


def test_stale_frame_warn_s_rejects_negative():
    with pytest.raises(ConfigError, match="stale_frame_warn_s"):
        ControllerConfig.parse({**BASE, "stale_frame_warn_s": -0.1})


def test_duration_warn_s_accepts_zero():
    # A warn-on-everything setting is an extreme but legitimate operator
    # choice (maximum verbosity while debugging), so 0.0 is not an error.
    cfg = ControllerConfig.parse({**BASE, "duration_warn_s": 0.0})
    assert cfg.duration_warn_s == 0.0


def test_duration_warn_s_rejects_non_number():
    with pytest.raises(ConfigError, match="duration_warn_s"):
        ControllerConfig.parse({**BASE, "duration_warn_s": "slow"})


# ---------------------------------------------------------------------------
# ConfigError hierarchy: there must be exactly one ConfigError type shared
# with vla.config_util, not a second module-local one.
# ---------------------------------------------------------------------------


def test_config_error_is_the_shared_config_util_type():
    from vla.config_util import ConfigError as SharedConfigError

    assert ConfigError is SharedConfigError


# ---------------------------------------------------------------------------
# arm_move_extra: the driver-facing "don't block until settled" flags.
# ---------------------------------------------------------------------------


def test_arm_move_extra_defaults_to_the_non_blocking_flags():
    cfg = ControllerConfig.parse(BASE)
    assert cfg.arm_move_extra == DEFAULT_ARM_MOVE_EXTRA
    assert cfg.arm_move_extra == {"wait": False, "waitAtEnd": False, "interpolate": False}


def test_arm_move_extra_default_is_a_copy_not_the_shared_module_dict():
    """Two controllers must not share one mutable default -- mutating one
    config's extra would otherwise change every other controller's, and the
    dict is handed to a driver that may do anything with it."""
    a = ControllerConfig.parse(BASE)
    b = ControllerConfig.parse(BASE)
    a.arm_move_extra["wait"] = "poisoned"
    assert b.arm_move_extra["wait"] is False
    assert DEFAULT_ARM_MOVE_EXTRA["wait"] is False


def test_arm_move_extra_replaces_the_default_wholesale():
    cfg = ControllerConfig.parse({**BASE, "arm_move_extra": {"direct": True}})
    assert cfg.arm_move_extra == {"direct": True}, "must not merge with the default"


def test_empty_arm_move_extra_is_distinct_from_absent():
    assert ControllerConfig.parse({**BASE, "arm_move_extra": {}}).arm_move_extra == {}


def test_arm_move_extra_rejects_a_non_object():
    for bad in ([1, 2], "wait", 3):
        with pytest.raises(ConfigError, match="arm_move_extra"):
            ControllerConfig.parse({**BASE, "arm_move_extra": bad})
