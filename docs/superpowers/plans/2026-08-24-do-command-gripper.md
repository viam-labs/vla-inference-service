# `do_command` Gripper Variant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fifth gripper variant that carries a VLA policy's gripper channel over a driver's `DoCommand` `{"get": true}` / `{"set": n}` pair, and fix two latency/correctness defects the design work uncovered.

**Architecture:** One new `GripperAdapter` subclass in the existing adapter module, selected by a new `gripper.type` string. The adapter normalizes between the driver's native scale and this module's `0.0 = fully open` convention using two required config endpoints, so a driver whose scale runs the opposite way needs no special case. Everything downstream (safety clamping, action-dim checks, the pre-flight probe, the tick read/write path) already branches on adapter *attributes* rather than type strings, so no control-flow changes are needed outside two one-line touch-ups.

**Tech Stack:** Python 3.12, `viam-sdk>=0.80.0`, pytest + pytest-asyncio (`asyncio_mode = "auto"`, so async tests need no decorator).

**Spec:** `docs/superpowers/specs/2026-08-24-do-command-gripper-design.md`

**Companion (different repo, not this plan):** `docs/superpowers/specs/2026-08-24-so101-nonblocking-set-handoff.md`

---

## Orientation for someone new to this codebase

Read these three things before Task 1. They explain why the tasks are shaped the way they are.

**What a gripper adapter is.** A VLA policy emits one continuous gripper value per tick alongside the arm joints. Viam has no single "gripper aperture" API, so `src/vla/controller/gripper.py` holds one adapter class per way of carrying that value onto real hardware, and `make_gripper_adapter()` picks one from config. Every adapter exposes `read() -> float` and `write(float)`, both in normalized `0.0`–`1.0` where **`0.0` is fully open** (the LeRobot dataset convention). The single exception is `ArmJointGripper`, which rides the arm's own joint vector in degrees and therefore has no `read`/`write` of its own.

**Why no downstream branching is needed.** Consumers key off four adapter attributes, never the config type string:

| Attribute | Meaning | Value for our new adapter |
| --- | --- | --- |
| `in_state` | value participates in the observation/action vector | `True` |
| `uses_degrees` | value is in degrees, not normalized | `False` |
| `dependency_name` | Viam resource this adapter needs injected | the configured `name` |
| `arm_joint_index` | set only when the channel rides the arm | `None` |

With those four values, `_build_safety` gives the channel the normalized `[0,1]` clamp, `_check_action_dim` expects `len(state_joint_indices) + 1`, `_preflight_gripper` probes it before any arm motion, and the tick path reads it via `gripper.read()` and writes it after the arm move. All confirmed against the code; do not add branching.

**The two type-string sites.** Only two places in `src/` compare the literal type string: `config.py:279` (`== "arm_joint"`, correctly needs no change) and `config.py:329` (a dependency tuple that **does** need our new name added, Task 7). `GRIPPER_TYPES` at `gripper.py:38` is the third place the string appears.

---

## File Structure

**Modified:**

| File | Responsibility | Tasks |
| --- | --- | --- |
| `tests/fakes.py` | Shared test doubles. Gains `FakeDoCommandGripper`; two existing fakes need defects fixed before dependent tests are writable. | 1, 3 |
| `src/vla/controller/gripper.py` | All gripper adapters + `make_gripper_adapter`. Gains `DoCommandGripper`, a shared inputs-reading helper, one `GRIPPER_TYPES` entry. | 2, 4, 5, 6, 10 |
| `src/vla/controller/config.py` | Config parsing/validation. One dependency-tuple entry. | 7 |
| `src/vla/controller/service.py` | Controller lifecycle + tick loop. One `extra=` kwarg, one docstring correction. | 8, 9 |
| `src/vla/controller/safety.py` | Clamping. Docstring only — it enumerates the normalized variants by name. | 10 |
| `README.md` | User-facing config reference. | 10 |

**Test files:** `tests/controller/test_gripper.py` (adapter unit tests), `tests/controller/test_config.py` (parse/dependency tests), `tests/controller/test_service.py` (integration through the tick loop).

No new source files. The adapter module is ~250 lines and cohesive; adding a sixth class keeps it well under the point where splitting would help.

**Test commands:**
- One test: `uv run pytest tests/controller/test_gripper.py::test_name -v`
- One file: `uv run pytest tests/controller/test_gripper.py -v`
- Full suite: `mise run test` (equivalent to `uv run pytest -m 'not integration and not differential' -v`)

---

## Task 1: Fix two test fakes that block later tasks

These are prerequisites, not cleanup. Task 2's tests and Task 9's test are both **unwritable** until these land: `FakeGripper` silently discards an empty `inputs` list, and `FakeArm` accepts the `extra` kwarg but throws it away.

**Files:**
- Modify: `tests/fakes.py:26-30` (`FakeArm.__init__`), `tests/fakes.py:37-45` (`FakeArm.move_to_joint_positions`), `tests/fakes.py:143` (`FakeGripper.__init__`)
- Test: `tests/controller/test_gripper.py`, `tests/controller/test_service.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/controller/test_gripper.py`:

```python
# ---------------------------------------------------------------------------
# fixture guards -- these two fakes silently swallowed the exact inputs the
# tests below depend on, so they are pinned here.
# ---------------------------------------------------------------------------


def test_fake_gripper_preserves_an_explicitly_empty_inputs_list():
    """A zero-DOF gripper model reports `[]`, which is a meaningful fixture.

    `list(inputs or [0.0])` collapsed it to `[0.0]`, making the empty-inputs
    refusal in `InputsGripper`/`ThresholdGripper` untestable.
    """
    assert FakeGripper(inputs=[]).inputs == []
    assert FakeGripper().inputs == [0.0]  # default unchanged
```

Add to the end of `tests/controller/test_service.py`:

```python
async def test_fake_arm_records_the_extra_it_was_called_with():
    """`extra` carries the driver-facing wait flag, so a fake that discards it
    cannot verify what the controller actually sent."""
    from viam.proto.component.arm import JointPositions

    arm = FakeArm(positions=[0.0] * 6)
    await arm.move_to_joint_positions(JointPositions(values=[0.0] * 6), extra={"wait": False})
    assert arm.move_extras == [{"wait": False}]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/controller/test_gripper.py::test_fake_gripper_preserves_an_explicitly_empty_inputs_list tests/controller/test_service.py::test_fake_arm_records_the_extra_it_was_called_with -v
```

Expected: both FAIL. The first with `assert [0.0] == []`; the second with `AttributeError: 'FakeArm' object has no attribute 'move_extras'`.

- [ ] **Step 3: Fix both fakes**

In `tests/fakes.py`, replace **only line 144** — the `self.inputs = ...` assignment inside `FakeGripper.__init__`. Leave the other four assignments (`supports_inputs`, `opened`, `grabbed`, `sent`) exactly as they are; deleting them breaks `test_gripper_incompatible_mode_is_caught_before_any_arm_motion` and every `open()`/`grab()`/`sent` assertion in `test_gripper.py`.

The method should read, in full, afterward:

```python
    def __init__(self, inputs=None, supports_inputs=True):
        # `if inputs is None`, not `inputs or [...]`: an explicitly empty list is
        # a meaningful fixture -- it is what a zero-DOF gripper model reports --
        # and `or` collapsed it to the default, hiding the case entirely.
        self.inputs = list([0.0] if inputs is None else inputs)
        self.supports_inputs = supports_inputs
        self.opened = 0
        self.grabbed = 0
        self.sent = []
```

In `tests/fakes.py`, `FakeArm.__init__`, add alongside `self.moves = []`:

```python
        self.move_extras = []
```

In `tests/fakes.py`, `FakeArm.move_to_joint_positions`, add immediately after `self.moves.append(positions)`:

```python
        self.move_extras.append(extra)
```

`StalledArm` (`tests/fakes.py:67`) overrides `move_to_joint_positions` wholesale, so add the same one-line append there too. No test asserts on it today, but leaving it out means a future `wait`-related test against a stalled arm silently sees an empty list instead of failing.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/controller/test_gripper.py::test_fake_gripper_preserves_an_explicitly_empty_inputs_list tests/controller/test_service.py::test_fake_arm_records_the_extra_it_was_called_with -v
```

Expected: 2 passed.

- [ ] **Step 5: Run the full suite — this task changes shared fixtures**

```bash
mise run test
```

Expected: all pass. A failure here means an existing test relied on `FakeGripper(inputs=[])` collapsing to `[0.0]`; fix that test rather than reverting the fake.

- [ ] **Step 6: Commit**

```bash
git add tests/fakes.py tests/controller/test_gripper.py tests/controller/test_service.py
git commit -m "test: stop FakeGripper and FakeArm discarding meaningful inputs"
```

---

## Task 2: Refuse an empty `get_current_inputs()` instead of reporting 0.0

`InputsGripper.read()` (`gripper.py:173-175`) and `ThresholdGripper.read()` (`gripper.py:199-201`) both do `float(values[0]) if values else 0.0`.

`get_current_inputs` is a **frame-system** interface, not an aperture channel: `framesystem.InputEnabled` (RDK `robot/framesystem/framesystem.go:36-43`) documents *"Input units are always in meters or radians"*, one value per kinematic DOF. `gripper.MakeModel` is documented as building *"a zero DoF Model"*, which is the common case — so the list is usually **empty**, and the current code reports a gripper that never moves, silently, forever.

**Files:**
- Modify: `src/vla/controller/gripper.py` (new helper; `InputsGripper.read`, `ThresholdGripper.read`)
- Test: `tests/controller/test_gripper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/controller/test_gripper.py`:

```python
@pytest.mark.parametrize(
    "block",
    [
        {"type": "gripper", "name": "g", "mode": "inputs"},
        {"type": "gripper", "name": "g", "mode": "threshold"},
    ],
    ids=["inputs", "threshold"],
)
async def test_zero_dof_gripper_read_refuses_instead_of_reporting_zero(block):
    """A zero-DOF gripper model reports no inputs at all, so there is no
    aperture to read. Reporting 0.0 looks like a gripper held fully open."""
    adapter = make_gripper_adapter(block, {"g": FakeGripper(inputs=[])})
    with pytest.raises(GripperRuntimeError, match="no kinematic DOF"):
        await adapter.read()


async def test_zero_dof_refusal_names_the_working_alternative():
    adapter = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "inputs"}, {"g": FakeGripper(inputs=[])}
    )
    with pytest.raises(GripperRuntimeError, match='gripper.type="do_command"'):
        await adapter.read()


async def test_nonempty_inputs_still_read_normally():
    adapter = make_gripper_adapter(
        {"type": "gripper", "name": "g", "mode": "inputs"}, {"g": FakeGripper(inputs=[0.25])}
    )
    assert await adapter.read() == pytest.approx(0.25)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/controller/test_gripper.py -k "zero_dof or nonempty_inputs" -v
```

Expected: the two `zero_dof` tests FAIL with `DID NOT RAISE`. `test_nonempty_inputs_still_read_normally` should already PASS — it is the regression guard.

- [ ] **Step 3: Add the shared helper and route both readers through it**

In `src/vla/controller/gripper.py`, add after `_clamp_unit` (line 96-97):

```python
async def _read_first_input(gripper: Any, name: str | None) -> float:
    """Read a gripper's aperture out of its frame-system inputs.

    ``get_current_inputs`` is a *frame system* interface, not an aperture
    channel: it returns one value per kinematic DOF, in radians or meters
    (``framesystem.InputEnabled`` in the RDK: "Input units are always in
    meters or radians"). ``gripper.MakeModel`` builds a zero-DOF model, so for
    most drivers the list is empty and carries no aperture at all. This used
    to fall back to 0.0, which reports a gripper permanently held fully open
    -- wrong, and invisible, for the entire life of the session.
    """
    values = await gripper.get_current_inputs()
    if not values:
        raise GripperRuntimeError(
            f"gripper {name!r} reports no kinematic DOF, so get_current_inputs() "
            "carries no aperture value; if the driver exposes a proportional "
            'DoCommand, use gripper.type="do_command"'
        )
    return float(values[0])
```

Then replace the body of `InputsGripper.read` **and** `ThresholdGripper.read` with:

```python
    async def read(self) -> float:
        return await _read_first_input(self._gripper, self.dependency_name)
```

Note the error text is deliberately conditional ("if the driver exposes..."). A driver implementing only `open()`/`grab()` with no proportional DoCommand has no working variant left after this change — its `threshold` *write* path was fine and only the read was broken. Pointing such an operator at `do_command` unconditionally would send them after something their driver cannot satisfy. See the spec's §3 for why a last-commanded-value fallback was rejected.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/controller/test_gripper.py -k "zero_dof or nonempty_inputs" -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full suite**

```bash
mise run test
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/vla/controller/gripper.py tests/controller/test_gripper.py
git commit -m "fix: refuse a zero-DOF gripper read instead of reporting 0.0 forever"
```

---

## Task 3: Add the `FakeDoCommandGripper` test double

**Files:**
- Modify: `tests/fakes.py` (append after `FakeGripper`)
- Test: `tests/controller/test_gripper.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/controller/test_gripper.py` (and add `FakeDoCommandGripper` to the `tests.fakes` import at the top of the file):

```python
# ---------------------------------------------------------------------------
# do_command
# ---------------------------------------------------------------------------


async def test_fake_do_command_gripper_round_trips():
    g = FakeDoCommandGripper(position=42.0)
    assert await g.do_command({"get": True}) == {"position": 42.0}
    await g.do_command({"set": 7.0})
    assert g.commands == [{"get": True}, {"set": 7.0}]
    assert await g.do_command({"get": True}) == {"position": 7.0}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/controller/test_gripper.py::test_fake_do_command_gripper_round_trips -v
```

Expected: FAIL with `ImportError: cannot import name 'FakeDoCommandGripper'`.

- [ ] **Step 3: Add the fake**

Append to `tests/fakes.py`:

```python
class FakeDoCommandGripper:
    """A gripper whose only proportional control is through ``DoCommand``.

    Mirrors the contract both `devrel:so101:gripper` and
    `viam:ufactory:gripper` implement: ``{"get": True}`` returns the current
    position under some key, ``{"set": n}`` commands a new one. Deliberately
    has no `get_current_inputs`/`go_to_inputs` -- that is the whole reason
    this variant exists, and omitting them keeps a test that reaches for the
    wrong API failing loudly.
    """

    def __init__(self, position=0.0, read_key="position", omit_read_key=False, read_value=None):
        self.position = position
        self.read_key = read_key
        # `omit_read_key` and `read_value` exist to drive the two malformed-response
        # paths: a driver whose key differs from the configured one, and a driver
        # returning a non-numeric value under the right key.
        self.omit_read_key = omit_read_key
        self.read_value = read_value
        self.commands = []

    async def do_command(self, command, **kwargs):
        self.commands.append(dict(command))
        if command.get("get") is True:
            if self.omit_read_key:
                return {"some_other_key": 1.0}
            value = self.position if self.read_value is None else self.read_value
            return {self.read_key: value}
        if "set" in command:
            self.position = command["set"]
            return {"position": self.position}
        raise AssertionError(f"unexpected command {command!r}")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/controller/test_gripper.py::test_fake_do_command_gripper_round_trips -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/fakes.py tests/controller/test_gripper.py
git commit -m "test: add a DoCommand-only gripper fake"
```

---

## Task 4: `DoCommandGripper` — config acceptance and required bounds

Build the class and the `make_gripper_adapter` branch, validation first. `read`/`write` bodies come in Tasks 5 and 6.

**Files:**
- Modify: `src/vla/controller/gripper.py:38` (`GRIPPER_TYPES`), `:43` (defaults), new class after `ThresholdGripper`, new branch in `make_gripper_adapter`
- Test: `tests/controller/test_gripper.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_do_command_adapter_wires_its_attributes():
    a = make_gripper_adapter(
        {"type": "do_command", "name": "grip", "open_value": 95.0, "closed_value": 0.0},
        {"grip": FakeDoCommandGripper()},
    )
    assert a.in_state is True
    assert a.uses_degrees is False
    assert a.dependency_name == "grip"
    assert a.arm_joint_index is None


@pytest.mark.parametrize("missing", ["open_value", "closed_value"])
def test_do_command_requires_both_bounds(missing):
    block = {"type": "do_command", "name": "grip", "open_value": 95.0, "closed_value": 0.0}
    del block[missing]
    with pytest.raises(GripperConfigError, match=missing):
        make_gripper_adapter(block, {"grip": FakeDoCommandGripper()})


def test_do_command_rejects_equal_bounds():
    """Equal endpoints make the read mapping a division by zero."""
    with pytest.raises(GripperConfigError, match="must differ"):
        make_gripper_adapter(
            {"type": "do_command", "name": "grip", "open_value": 50.0, "closed_value": 50.0},
            {"grip": FakeDoCommandGripper()},
        )


def test_do_command_requires_a_name():
    with pytest.raises(GripperConfigError, match="name"):
        make_gripper_adapter(
            {"type": "do_command", "open_value": 95.0, "closed_value": 0.0}, {}
        )


def test_do_command_rejects_close_threshold():
    """`close_threshold` belongs only to gripper/threshold."""
    with pytest.raises(GripperConfigError, match="close_threshold"):
        make_gripper_adapter(
            {
                "type": "do_command",
                "name": "grip",
                "open_value": 95.0,
                "closed_value": 0.0,
                "close_threshold": 0.5,
            },
            {"grip": FakeDoCommandGripper()},
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/controller/test_gripper.py -k "do_command" -v
```

Expected: the five new tests FAIL. `test_do_command_adapter_wires_its_attributes` fails with `gripper.type must be one of (...)`.

- [ ] **Step 3: Implement**

In `src/vla/controller/gripper.py`, extend `GRIPPER_TYPES` (line 38) and add a default:

```python
GRIPPER_TYPES = ("arm_joint", "servo", "gripper", "do_command", "none")
```

```python
_DEFAULT_READ_KEY = "position"
```

Add the class after `ThresholdGripper`:

```python
class DoCommandGripper(GripperAdapter):
    """Proportional control through ``DoCommand``, for drivers that expose it
    there rather than through the typed API.

    Contract: any gripper whose ``DoCommand`` implements ``{"get": True}`` ->
    ``{read_key: number}`` and ``{"set": number}``. Both
    ``devrel:so101:gripper`` and ``viam:ufactory:gripper`` do, with the same
    request shape -- they differ only in the read response key and the value
    range.

    ``open_value``/``closed_value`` are the driver's own native values at the
    two extremes: percent for so-101 (95 / 0), raw units for xarm (840 / 2).
    Expressing the mapping in terms of the two *named endpoints* rather than a
    min/max ordering is what lets it carry a driver whose scale runs opposite
    to this module's ``0.0 = fully open`` convention -- which both of those
    drivers do, since for them a higher number means more open.
    """

    def __init__(
        self,
        name: str,
        gripper: Any,
        open_value: float,
        closed_value: float,
        read_key: str,
        write_args: Mapping[str, Any],
    ) -> None:
        if open_value == closed_value:
            raise GripperConfigError(
                "gripper.open_value and gripper.closed_value must differ, both are "
                f"{open_value!r}"
            )
        self.dependency_name = name
        self._gripper = gripper
        self._open = open_value
        self._closed = closed_value
        self._read_key = read_key
        self._write_args = dict(write_args)
```

Add the branch in `make_gripper_adapter`, immediately after the `if kind == "servo":` block. `name` is already resolved and validated at that point.

```python
    if kind == "do_command":
        for field in ("open_value", "closed_value"):
            if field not in raw:
                raise GripperConfigError(
                    f'gripper.type="do_command" requires {field}; it is the driver-native '
                    "value at that extreme (so-101: 95/0 percent, xarm: 840/2 raw units). "
                    "There is no safe default -- a percentage guess silently saturates a "
                    "raw-unit driver in the first percent of its travel."
                )
        open_value = as_float(raw["open_value"], "gripper.open_value")
        closed_value = as_float(raw["closed_value"], "gripper.closed_value")
        read_key = as_str(raw.get("read_key", _DEFAULT_READ_KEY), "gripper.read_key")
        write_args = raw.get("write_args") or {}
        return DoCommandGripper(
            name, dependencies.get(name), open_value, closed_value, read_key, write_args
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/controller/test_gripper.py -k "do_command" -v
```

Expected: 7 passed — four unparametrized new tests, `test_do_command_requires_both_bounds` ×2, and the Task 3 fake test.

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/gripper.py tests/controller/test_gripper.py
git commit -m "feat: accept gripper.type=do_command with required scale endpoints"
```

---

## Task 5: `DoCommandGripper.read()` — normalization and clamping

**Files:**
- Modify: `src/vla/controller/gripper.py` (`DoCommandGripper.read`)
- Test: `tests/controller/test_gripper.py`

- [ ] **Step 1: Write the failing tests**

```python
def _do_cmd(gripper, **overrides):
    block = {
        "type": "do_command",
        "name": "grip",
        "open_value": 95.0,
        "closed_value": 0.0,
        **overrides,
    }
    return make_gripper_adapter(block, {"grip": gripper})


@pytest.mark.parametrize(
    "raw,expected",
    [(95.0, 0.0), (0.0, 1.0), (47.5, 0.5)],
    ids=["open-rail", "closed-rail", "midpoint"],
)
async def test_do_command_read_normalizes_an_inverted_scale(raw, expected):
    """so-101 counts *up* toward open; this module's 0.0 means fully open."""
    adapter = _do_cmd(FakeDoCommandGripper(position=raw))
    assert await adapter.read() == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,expected", [(840.0, 0.0), (2.0, 1.0), (421.0, 0.5)], ids=["open", "closed", "mid"]
)
async def test_do_command_read_handles_xarm_raw_units(raw, expected):
    adapter = _do_cmd(
        FakeDoCommandGripper(position=raw, read_key="pos"),
        read_key="pos",
        open_value=840.0,
        closed_value=2.0,
    )
    assert await adapter.read() == pytest.approx(expected)


async def test_do_command_read_clamps_past_the_open_rail():
    """Not defensive: so-101's openPosition is 95 but the servo travels to 100,
    so an ordinary reading of 98 maps to -0.03 unclamped and puts an
    out-of-range value into the state vector the policy sees."""
    adapter = _do_cmd(FakeDoCommandGripper(position=98.0))
    assert await adapter.read() == 0.0


async def test_do_command_read_clamps_past_the_closed_rail():
    adapter = _do_cmd(FakeDoCommandGripper(position=-5.0))
    assert await adapter.read() == 1.0


async def test_do_command_read_uses_the_configured_read_key():
    adapter = _do_cmd(FakeDoCommandGripper(position=95.0, read_key="pos"), read_key="pos")
    assert await adapter.read() == pytest.approx(0.0)


async def test_do_command_read_errors_when_the_key_is_absent():
    adapter = _do_cmd(FakeDoCommandGripper(omit_read_key=True))
    with pytest.raises(GripperRuntimeError, match="some_other_key"):
        await adapter.read()


async def test_do_command_read_errors_on_a_non_numeric_value():
    adapter = _do_cmd(FakeDoCommandGripper(read_value="halfway"))
    with pytest.raises(GripperRuntimeError, match="non-numeric"):
        await adapter.read()


async def test_do_command_read_rejects_a_bool_as_non_numeric():
    """`isinstance(True, int)` is True in Python, so bool needs excluding
    explicitly or a driver returning True reads as fully closed."""
    adapter = _do_cmd(FakeDoCommandGripper(read_value=True))
    with pytest.raises(GripperRuntimeError, match="non-numeric"):
        await adapter.read()


async def test_do_command_read_emits_the_get_command():
    g = FakeDoCommandGripper(position=95.0)
    await _do_cmd(g).read()
    assert g.commands == [{"get": True}]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/controller/test_gripper.py -k "do_command_read" -v
```

Expected: all FAIL. The value tests raise `NotImplementedError` (inherited from `GripperAdapter.read`); the three error-path tests fail because `pytest.raises(GripperRuntimeError)` does not catch that `NotImplementedError`.

- [ ] **Step 3: Implement `read`**

Add to `DoCommandGripper`:

```python
    async def read(self) -> float:
        res = await self._gripper.do_command({"get": True})
        if not isinstance(res, Mapping) or self._read_key not in res:
            got = sorted(res) if isinstance(res, Mapping) else res
            raise GripperRuntimeError(
                f"gripper {self.dependency_name!r} do_command({{'get': True}}) returned "
                f"no {self._read_key!r} key; got {got!r}. Set gripper.read_key to the key "
                "this driver actually returns."
            )
        value = res[self._read_key]
        # `isinstance(True, int)` is True, so bool has to be excluded by hand or a
        # driver returning a flag under this key reads as a legitimate position.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GripperRuntimeError(
                f"gripper {self.dependency_name!r} returned a non-numeric "
                f"{self._read_key!r}: {value!r} ({type(value).__name__})"
            )
        # Clamped because the endpoints are a *calibration*, not a hard travel
        # limit: so-101 calls 95 fully open while the servo reaches 100.
        return _clamp_unit((float(value) - self._open) / (self._closed - self._open))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/controller/test_gripper.py -k "do_command_read" -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/gripper.py tests/controller/test_gripper.py
git commit -m "feat: read a do_command gripper's position onto the normalized scale"
```

---

## Task 6: `DoCommandGripper.write()` and the `write_args` guards

The two `write_args` validations here are load-bearing. The write merges as `{"set": raw, **write_args}`, so a `set` key inside `write_args` would silently **replace** the setpoint computed from the policy's action — the gripper would park at a constant with no error anywhere.

**Files:**
- Modify: `src/vla/controller/gripper.py` (`DoCommandGripper.write`; validation in `make_gripper_adapter`)
- Test: `tests/controller/test_gripper.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(
    "value,expected", [(0.0, 95.0), (1.0, 0.0), (0.5, 47.5)], ids=["open", "closed", "mid"]
)
async def test_do_command_write_maps_onto_the_driver_scale(value, expected):
    g = FakeDoCommandGripper()
    await _do_cmd(g).write(value)
    assert g.commands == [{"set": pytest.approx(expected)}]


@pytest.mark.parametrize("value,expected", [(-0.2, 95.0), (1.7, 0.0)])
async def test_do_command_write_clamps_out_of_range_actions(value, expected):
    g = FakeDoCommandGripper()
    await _do_cmd(g).write(value)
    assert g.commands == [{"set": pytest.approx(expected)}]


async def test_do_command_write_merges_write_args():
    g = FakeDoCommandGripper()
    await _do_cmd(g, write_args={"wait": False}).write(1.0)
    assert g.commands == [{"set": pytest.approx(0.0), "wait": False}]


async def test_do_command_write_omits_extras_by_default():
    g = FakeDoCommandGripper()
    await _do_cmd(g).write(1.0)
    assert list(g.commands[0]) == ["set"]


def test_do_command_rejects_a_set_key_in_write_args():
    """A `set` entry would override the computed setpoint and park the gripper
    at a constant, with nothing raised anywhere."""
    with pytest.raises(GripperConfigError, match="write_args"):
        _do_cmd(FakeDoCommandGripper(), write_args={"set": 12.0})


def test_do_command_rejects_non_mapping_write_args():
    with pytest.raises(GripperConfigError, match="write_args"):
        _do_cmd(FakeDoCommandGripper(), write_args=[("wait", False)])


async def test_do_command_round_trips_through_the_driver():
    """write() then read() should land back on the value written."""
    g = FakeDoCommandGripper()
    adapter = _do_cmd(g)
    await adapter.write(0.25)
    assert await adapter.read() == pytest.approx(0.25)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/controller/test_gripper.py -k "do_command_write or write_args or round_trips_through" -v
```

Expected: the write tests FAIL with `NotImplementedError`; the two validation tests FAIL with `DID NOT RAISE`.

- [ ] **Step 3: Implement `write` and the guards**

Add to `DoCommandGripper`:

```python
    async def write(self, value: float) -> None:
        raw = self._open + _clamp_unit(value) * (self._closed - self._open)
        await self._gripper.do_command({"set": raw, **self._write_args})
```

In `make_gripper_adapter`'s `do_command` branch, replace `write_args = raw.get("write_args") or {}` with:

```python
        write_args = raw.get("write_args") or {}
        if not isinstance(write_args, Mapping):
            raise GripperConfigError(
                f"gripper.write_args must be an object, got {write_args!r}"
            )
        if "set" in write_args:
            raise GripperConfigError(
                'gripper.write_args must not contain "set": it is merged into the set '
                "command and would silently replace the setpoint computed from the "
                "policy's action, parking the gripper at a constant"
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/controller/test_gripper.py -v
```

Expected: the whole file passes.

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/gripper.py tests/controller/test_gripper.py
git commit -m "feat: write a do_command gripper setpoint, guarding write_args"
```

---

## Task 7: Register the dependency in config parsing

Without this the configured resource is never requested, so `dependencies.get(name)` hands the adapter `None` and every call fails with `AttributeError` at runtime.

**Files:**
- Modify: `src/vla/controller/config.py:329`
- Test: `tests/controller/test_config.py`

> The spec's §5 lists the `write_args` and bounds rejections under `test_config.py`; this plan puts them in `test_gripper.py` (Tasks 4 and 6) instead. That is deliberate, not dropped coverage: `ControllerConfig.parse` validates only `gripper.type` against `GRIPPER_TYPES`, and every other gripper field is validated inside `make_gripper_adapter` — which is where the spec's §1 says these surface.

- [ ] **Step 1: Write the failing tests**

```python
def test_do_command_gripper_adds_dependency():
    cfg = ControllerConfig.parse(
        {
            **BASE,
            "gripper": {
                "type": "do_command",
                "name": "grip",
                "open_value": 95.0,
                "closed_value": 0.0,
            },
        }
    )
    assert "grip" in cfg.dependencies()
```

And extend the existing `@pytest.mark.parametrize` at `tests/controller/test_config.py:225` to cover the new type:

```python
@pytest.mark.parametrize("kind", ["arm_joint", "servo", "gripper", "do_command", "none"])
def test_accepts_every_known_gripper_type(kind):
    extra = (
        {"joint_index": 5}
        if kind == "arm_joint"
        else {"name": "g", "open_value": 95.0, "closed_value": 0.0}
        if kind == "do_command"
        else {"name": "g"}
        if kind != "none"
        else {}
    )
    cfg = ControllerConfig.parse({**BASE, "gripper": {"type": kind, **extra}})
    assert cfg.gripper["type"] == kind
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/controller/test_config.py -k "do_command or every_known_gripper_type" -v
```

Expected: `test_do_command_gripper_adds_dependency` FAILS (`'grip' not in [...]`). The parametrized case for `do_command` should already PASS — `ControllerConfig.parse` only validates the type string against `GRIPPER_TYPES`, which Task 4 already extended. Keep it as a guard.

- [ ] **Step 3: Add the type to the dependency tuple**

`src/vla/controller/config.py:329`:

```python
        if self.gripper.get("type") in ("servo", "gripper", "do_command") and name:
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/controller/test_config.py -v
```

Expected: the whole file passes.

- [ ] **Step 5: Commit**

```bash
git add src/vla/controller/config.py tests/controller/test_config.py
git commit -m "feat: resolve the do_command gripper's resource dependency"
```

---

## Task 8: Verify the variant through the real tick loop

Nothing here should need production changes — this task proves the "no downstream branching" claim. If a test fails, fix the adapter's attributes rather than adding a branch.

**Files:**
- Modify: `tests/controller/test_service.py:30` (import line)
- Test: `tests/controller/test_service.py`

**Harness in this file** (read `tests/controller/test_service.py:111-155` before starting): `_config(**overrides)` builds a `ServiceConfig`, `_deps(policy=None, arm=None, camera=None, **extra)` builds the dependency dict (extra kwargs become named resources), `_svc(config, deps)` returns a reconfigured `VLAController`, and `_wait_for_state(svc, *states)` polls status until one is reached. There is **no** one-shot tick helper — tests drive the loop with `start` → `asyncio.sleep(...)` → `stop`. Do not add a helper; follow the existing pattern.

Note `status["last_error"]` is `""` when there is no error, never `None` (`service.py:195` does `self._last_error or ""`).

- [ ] **Step 1: Extend the import**

`tests/controller/test_service.py:30`:

```python
from tests.fakes import FakeArm, FakeCamera, FakeDoCommandGripper, FakeGripper, StalledArm
```

- [ ] **Step 2: Write the tests**

Modeled on `test_gripper_write_happens_after_arm_move_for_non_arm_joint_gripper` (line 507) and `test_gripper_incompatible_mode_is_caught_before_any_arm_motion` (line 566).

```python
async def test_do_command_gripper_is_driven_through_a_real_tick():
    """5 driven joints + 1 gripper channel accounted for, and the gripper's
    own component actually receives the policy's action."""
    arm = FakeArm(positions=[0.0] * 5)
    g = FakeDoCommandGripper(position=95.0)
    policy = FakePolicyClient(action_dim=6, action_value=1.0)
    svc = _svc(
        config=_config(
            gripper={
                "type": "do_command",
                "name": "g",
                "open_value": 95.0,
                "closed_value": 0.0,
            }
        ),
        deps=_deps(policy=policy, arm=arm, g=g),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    # Status BEFORE stop, and via its own command: `stop` returns {"ok": True}
    # (service.py:171-172), so reading last_error off its result is a KeyError.
    # This is the established pattern -- see test_service.py:455.
    status = await svc.do_command({"command": "status"})
    await svc.do_command({"command": "stop"})
    assert status["last_error"] == ""
    # A policy action of 1.0 is fully closed, which on this driver's scale is 0.0.
    sets = [c["set"] for c in g.commands if "set" in c]
    assert sets, g.commands
    assert sets[-1] == pytest.approx(0.0)


async def test_do_command_gripper_write_order_is_after_arm_move_not_before():
    """Same ordering guarantee the other non-arm_joint variants have."""
    events = []
    arm = FakeArm(positions=[0.0] * 5)
    g = FakeDoCommandGripper(position=95.0)
    policy = FakePolicyClient(action_dim=6, action_value=1.0)

    real_move = arm.move_to_joint_positions
    real_do = g.do_command

    async def tracked_move(*a, **kw):
        events.append("arm")
        return await real_move(*a, **kw)

    async def tracked_do(command, **kw):
        if "set" in command:
            events.append("gripper")
        return await real_do(command, **kw)

    arm.move_to_joint_positions = tracked_move
    g.do_command = tracked_do

    svc = _svc(
        config=_config(
            gripper={
                "type": "do_command",
                "name": "g",
                "open_value": 95.0,
                "closed_value": 0.0,
            }
        ),
        deps=_deps(policy=policy, arm=arm, g=g),
    )
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    # events[0] is the pre-flight probe's own write, before any arm motion.
    first_arm_idx = events.index("arm")
    assert events[first_arm_idx + 1] == "gripper", events


async def test_do_command_malformed_read_is_caught_before_any_arm_motion():
    """`_preflight_gripper` reads then writes back, so a driver whose response
    key does not match `read_key` fails before the arm is ever commanded."""
    arm = FakeArm(positions=[0.0] * 5)
    g = FakeDoCommandGripper(omit_read_key=True)
    # action_dim=6 matters: with 5 state_joint_indices and in_state=True the
    # expected dim is 6, and a mismatch would raise in _check_action_dim
    # *before* the preflight probe -- passing the arm-motion assertion for
    # entirely the wrong reason.
    policy = FakePolicyClient(action_dim=6)
    svc = _svc(
        config=_config(
            gripper={
                "type": "do_command",
                "name": "g",
                "open_value": 95.0,
                "closed_value": 0.0,
            }
        ),
        deps=_deps(policy=policy, arm=arm, g=g),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "some_other_key" in status["last_error"]
    assert len(arm.moves) == 0, "the arm must never move before the gripper probe fails"
    assert arm.stopped >= 1


async def test_zero_dof_inputs_gripper_is_caught_before_any_arm_motion():
    """The Task 2 refusal has to surface at *startup*, not mid-episode -- that
    is the whole claim the pre-flight probe makes. Adapter-level tests in
    test_gripper.py cannot prove it."""
    arm = FakeArm(positions=[0.0] * 5)
    gripper = FakeGripper(inputs=[])
    policy = FakePolicyClient(action_dim=6)
    svc = _svc(
        config=_config(gripper={"type": "gripper", "name": "g", "mode": "inputs"}),
        deps=_deps(policy=policy, arm=arm, g=gripper),
    )
    await svc.do_command({"command": "start", "task": "t"})
    status = await _wait_for_state(svc, "error")
    assert "no kinematic DOF" in status["last_error"]
    assert len(arm.moves) == 0, "the arm must never move before the gripper probe fails"
    assert arm.stopped >= 1
```

- [ ] **Step 3: Run the tests**

```bash
uv run pytest tests/controller/test_service.py -k "do_command or zero_dof" -v
```

Expected: 4 passed, with no production changes. If the first test fails on action-dim, check `in_state` is `True`. If a preflight test sees arm motion, check `arm_joint_index` is `None`.

- [ ] **Step 4: Commit**

```bash
git add tests/controller/test_service.py
git commit -m "test: cover the do_command gripper and zero-DOF refusal through the loop"
```

---

## Task 9: Stop blocking on arm settle every tick

`service.py:660` calls `move_to_joint_positions` with no `extra`. so-101's `parseWaitExtra` (`components/arm/motion.go:242-249`) defaults to `true`, so the arm blocks until every servo settles — on every tick, against a 100 ms budget at 10 fps. A VLA supersedes each setpoint on the next tick, so waiting for the previous one to finish defeats the chunked-action design.

No config flag: `extra` is a free-form struct and drivers ignore keys they do not read, so this is inert on other arms.

**Files:**
- Modify: `src/vla/controller/service.py:660`
- Test: `tests/controller/test_service.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_arm_move_asks_the_driver_not_to_wait_for_settle():
    """A driver that blocks until the arm settles burns the whole tick budget
    waiting for a setpoint the next tick is about to replace. Drivers that do
    not read `wait` ignore it."""
    arm = FakeArm(positions=[0.0] * 6)
    svc = _svc(config=_config(), deps=_deps(arm=arm))
    await svc.do_command({"command": "start", "task": "t"})
    await asyncio.sleep(0.15)
    await svc.do_command({"command": "stop"})
    assert arm.move_extras, "the arm was never commanded"
    assert all(e == {"wait": False} for e in arm.move_extras), arm.move_extras
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/controller/test_service.py::test_arm_move_asks_the_driver_not_to_wait_for_settle -v
```

Expected: FAIL — `move_extras` contains `None`.

- [ ] **Step 3: Pass the extra**

`src/vla/controller/service.py:660`:

```python
            # `wait: False` because the next tick supersedes this setpoint: a driver
            # that blocks until the arm physically settles (so-101's default) spends
            # the entire tick budget waiting for a target we are about to replace.
            # Free-form `extra`, so a driver that does not read the key ignores it.
            await arm.move_to_joint_positions(
                JointPositions(values=target), extra={"wait": False}
            )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest tests/controller/test_service.py::test_arm_move_asks_the_driver_not_to_wait_for_settle -v
```

Expected: 1 passed.

- [ ] **Step 5: Run the full suite**

```bash
mise run test
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/vla/controller/service.py tests/controller/test_service.py
git commit -m "perf: stop waiting for arm settle on every control tick"
```

---

## Task 10: Documentation

Four sites, all of which currently state something now false or incomplete. No tests.

**Files:**
- Modify: `src/vla/controller/gripper.py:1-26` (module docstring), `src/vla/controller/safety.py:27-31`, `src/vla/controller/service.py:414-419` (`_preflight_gripper` docstring), `README.md:436-480` (§"Gripper variants")

- [ ] **Step 1: Correct the adapter module docstring**

`gripper.py:1-26` says "Viam offers three components" and describes `inputs` as "normalized 0..1". Update the count, add the `do_command` variant, and correct the `inputs` description: `get_current_inputs`/`go_to_inputs` are a frame-system interface carrying one value per kinematic DOF in radians or meters, so those variants work only against a driver whose gripper model is *jointed*; most gripper models, `devrel:so101:gripper` included, are zero-DOF. Keep the closing unit-convention paragraph, extending it to note `do_command` is normalized like the others.

- [ ] **Step 2: Add the variant to the safety docstring**

`safety.py:27-31` enumerates the variants that skip the degree clamps by name. Add `do_command`:

```
The degrees-based clamps (delta and limit) skip a normalized gripper
channel (`servo`, `gripper/inputs`, `gripper/threshold`, `do_command`): that
channel is 0.0-1.0, so a degree-shaped limit would either never fire
(useless) or fire constantly on ordinary gripper motion (worse than useless).
```

- [ ] **Step 3: Correct the pre-flight docstring**

`service.py:414-419` claims the probe is "a no-op for `InputsGripper`/`ServoGripper` (they land on the same value already reported)". That is now untrue for `DoCommandGripper`: with `open_value=95` and a raw reading of 98, the read clamps to 0.0 and the write-back commands the jaw to 95. Add one sentence saying so, and noting it is the same small, deliberate actuation `ThresholdGripper` already performs at preflight.

- [ ] **Step 4: Add the README entry**

First fix `README.md:438-439`, which says "Viam offers three different **components** that can carry it" — now five variants across four components. Same correction as the module docstring in Step 1.

Then, in §"Gripper variants" (`README.md:436-480`), add after the `gripper`/`threshold` entry:

````markdown
**`do_command`** — proportional control for drivers that expose it through
`DoCommand` rather than the typed API. Requires that the driver implement
`{"get": true}` → `{<read_key>: number}` and `{"set": number}`.
`devrel:so101:gripper` and `viam:ufactory:gripper` both do.

`open_value` and `closed_value` are the driver's own native values at each
extreme and are **required** — there is no safe default, since a percentage
guess would silently saturate a raw-unit driver within the first percent of
its travel. They may run in either direction, which is how this variant
carries a driver whose scale counts *up* toward open.

```json
{ "type": "do_command", "name": "grip",
  "open_value": 95.0, "closed_value": 0.0,
  "write_args": { "wait": false } }
```

```json
{ "type": "do_command", "name": "grip", "read_key": "pos",
  "open_value": 840.0, "closed_value": 2.0 }
```

`read_key` defaults to `"position"`. `write_args` is merged into the `set`
command and defaults to `{}`; it may not contain a `set` key.

> The xarm config above is functionally correct but impractical in a 10 fps
> loop: that driver polls until the jaw settles, up to 10 seconds, with no
> way to opt out.
````

Then rewrite the `inputs` and `threshold` entries per Step 1's correction, and extend the closing sentence at `README.md:478` to cover five variants. That sentence — "Except for `arm_joint`, every variant's value is normalized `0.0`–`1.0`" — stays **true**: it describes what each adapter hands the controller, whereas the `inputs` correction is about what the driver hands the adapter. State that distinction explicitly in the `inputs` entry so the next reader does not conflate the two, as the design spec's own sections initially did.

Finally, add the note the spec's §3 asks for (its closing paragraph): `InputsGripper.write` catches only `NotImplementedError`, but a Go driver's `errors.ErrUnsupported` arrives in Python as a `GRPCError`, so the "reconfigure to threshold" hint never fires for one. Task 2's empty-inputs guard makes the *read* fail first, which is the path that actually runs — so this is a documentation note in the `inputs` entry, not a code change.

- [ ] **Step 5: Verify the docs match the code**

```bash
mise run test
uv run python -c "
from vla.controller.gripper import GRIPPER_TYPES
print(GRIPPER_TYPES)
assert 'do_command' in GRIPPER_TYPES
"
```

Then re-read each of the four edited passages against the implementation. Every config key named in the README must exist in `make_gripper_adapter`, and every key it accepts must appear in the README.

- [ ] **Step 6: Commit**

```bash
git add src/vla/controller/gripper.py src/vla/controller/safety.py src/vla/controller/service.py README.md
git commit -m "docs: document do_command, and correct the inputs unit claim"
```

---

## Definition of done

- [ ] `mise run test` passes
- [ ] `gripper.type="do_command"` drives `devrel:so101:gripper` end to end with `open_value: 95.0, closed_value: 0.0`
- [ ] A zero-DOF gripper under `inputs` refuses at startup naming `do_command`, with no arm motion first (Task 8's `test_zero_dof_inputs_gripper_is_caught_before_any_arm_motion`); the `threshold` read shares the same helper and is covered at adapter level in Task 2
- [ ] The arm receives `extra={"wait": False}` on every tick
- [ ] Outside `make_gripper_adapter`'s own dispatch, no branch keys off the `do_command` type string except `GRIPPER_TYPES` and `config.py:329`
- [ ] README documents every config key the adapter accepts, and accepts every key it documents

## Not in this plan

- The so-101 module change that makes the gripper write non-blocking. Separate repo, separate session — see the companion handoff doc. This plan is complete and shippable without it; the variant works, at up to 2 s per gripper write, until it lands. Then `write_args: {"wait": false}` starts taking effect with no code change here.
- A 1-DOF gripper kinematic model in so-101, which is what `inputs` would need to be genuinely usable. Analyzed and rejected in the spec's "Out of scope".
