# `do_command` gripper variant, and two latency/correctness fixes

Date: 2026-08-24
Status: design approved, not yet implemented
Companion doc: `docs/superpowers/specs/2026-08-24-so101-nonblocking-set-handoff.md`
(the so-101 module change this design depends on for acceptable tick latency)

## Motivation

Running this module against `lerobot/smolvla_base` with an SO-ARM101 (the
`devrel:so101-arm` module, `viam-devrel/so-101`) surfaced that **none** of the
four existing `gripper.type` variants can carry the policy's gripper channel
onto `devrel:so101:gripper`:

| Variant | Why it fails against `devrel:so101:gripper` |
| --- | --- |
| `arm_joint` | `devrel:so101:arm` drives servos 1–5 only. There is no joint index 5 in its joint vector, so `_check_joint_indices` (`src/vla/controller/service.py:399`) refuses at startup. Widening the arm's `servo_ids` to `[1..6]` is not available either: `components/arm/arm.go:136-140` rejects any ID outside 1–5 outright (`"arm servo IDs must be 1-5, got %d"`), so the arm never constructs. The embedded kinematic model is 5-DOF regardless (`internal/geometry/so101.json` declares joints 1–5). |
| `servo` | The so-101 module registers no `rdk:component:servo` model. Servo 6 is reachable only through DoCommand. |
| `gripper` / `inputs` | `CurrentInputs` and `GoToInputs` both `return errors.ErrUnsupported` (`components/gripper/gripper.go:497-502`). The pre-flight probe fails on the *read*, before `write` is reached. |
| `gripper` / `threshold` | Same — `ThresholdGripper.read()` also goes through `get_current_inputs()`. `open()`/`grab()` work; the read side does not. |

What `devrel:so101:gripper` *does* expose is proportional control, via
DoCommand rather than the typed API (`components/gripper/gripper.go:366-387`):

- read: `{"get": true}` → `{"position": <float percent>}`
- write: `{"set": <float percent>}` → `{"position": <float percent>}`

`viam:ufactory:gripper` (`viam-ufactory-xarm/arm/gripper.go:383-402`) exposes
the *same request shape*, differing only in the **read** response key (`pos`,
at `:389`) and the value range (raw units, roughly 0–850, versus 0–100
percent). One variant therefore covers both drivers with no templating and no
per-driver presets.

Only the read key differs, because only the read key is consumed: xarm's *write*
returns `{"position": …}` (`:401`), the same key so-101 uses, and both are
discarded.

## Contract

> Any gripper whose `DoCommand` implements `{"get": true}` → `{<read_key>: number}`
> and `{"set": number}`.

That sentence is the whole contract. It is narrow enough to validate and wide
enough to cover the two drivers we have in hand.

## 1. Config surface

A fifth entry in `GRIPPER_TYPES` (`src/vla/controller/gripper.py`):

```json
{
  "type": "do_command",
  "name": "grip",
  "open_value": 95.0,
  "closed_value": 0.0,
  "read_key": "position",
  "write_args": {}
}
```

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `name` | yes | — | Viam resource name; becomes a module dependency. |
| `open_value` | **yes** | — | The driver-native value at fully open. |
| `closed_value` | **yes** | — | The driver-native value at fully closed. |
| `read_key` | no | `"position"` | Key to pull from the `{"get": true}` response. |
| `write_args` | no | `{}` | Extra pairs merged into the `set` command. Must be a mapping, and must not contain `get` or `set` — the adapter's own protocol keys. |

`open_value` and `closed_value` are **required, not defaulted**. A percentage
default (`100`/`0`) would silently saturate a 0–850 driver like xarm at the
first percent of its travel — wrong but plausible, and invisible at runtime.
Requiring them makes a mismatched range a `ConfigError` instead. Equal values
are also rejected (division by zero in the read mapping).

`write_args` defaults to `{}` so no driver-specific behavior is baked into
this module. The so-101 README example sets `{"wait": false}` once the
companion so-101 change lands; xarm ignores an unrecognized key, as does any
driver that does not read it.

Two validations on `write_args` are load-bearing rather than defensive. It must
be rejected if it is not a mapping, and rejected if it contains either of the
adapter's own protocol keys, `get` or `set`. The write merges as
`{"set": raw, **write_args}`, so:

- a `set` entry silently *replaces* the setpoint just computed from the policy's
  action — the gripper parks at a constant;
- a `get` entry makes any driver that checks `get` before `set` treat every
  write as a read — the gripper never moves at all. This is not hypothetical:
  the project's own `FakeDoCommandGripper` checks in that order, and so does
  `devrel:so101:gripper` (`components/gripper/gripper.go:366-387`).

Both failures are invisible at runtime — the gripper simply stops tracking the
policy with nothing raised — so both have to be config errors.

The mapping check must also come *first*, and `write_args` must default via
`raw.get("write_args", {})` rather than `or {}`: `or` folds falsy non-mappings
(`[]`, `0`, `False`, `''`) into `{}` before either guard can see them, and
`dict([])`/`dict('')` both succeed, so those would be accepted silently.

**On timing:** these are `ConfigError`s, but they do not surface at
`validate_config`/`reconfigure`. `make_gripper_adapter` runs inside `_run()`
(`service.py:454`), *after* `_await_policy()`, so a bad gripper block surfaces
on the `start` command and potentially after a long policy wait. This matches
how `servo`'s `min_deg`/`max_deg` behave today, so it is consistent — but the
implementation plan must not promise config-time validation.

**Rejected: a `timeout_s` key.** Gripper write failures are unconditionally
fatal by design (`service.py:656-662`), and a timed-out DoCommand leaves the
jaw mid-move. It would trade a stalled tick for a killed session. The existing
`duration_warn_s` already surfaces slow ticks.

**Rejected: `$value` templating and a preset registry.** Both were designed
against a wrong reading of so-101's DoCommand — the `cmd["command"]` switch is
only reached *after* the un-nested `get`/`set` checks. With the request shape
uniform across both drivers, neither mechanism earns its complexity.

## 2. `DoCommandGripper`

New adapter class in `src/vla/controller/gripper.py`, alongside the five that
already live there (`NoGripper`, `ArmJointGripper`, `ServoGripper`,
`InputsGripper`, `ThresholdGripper` — five classes serving four type strings,
because `gripper` dispatches on `mode`).

```
read():
    res = await self._gripper.do_command({"get": True})
    raw = res[read_key]                       # missing/non-numeric -> GripperRuntimeError
    return clamp01((raw - open_value) / (closed_value - open_value))

write(v):
    raw = open_value + clamp01(v) * (closed_value - open_value)
    await self._gripper.do_command({"set": raw, **write_args})
```

Direction inversion falls out of the mapping for free. so-101 reports
`open=95, closed=0` — higher is more open — which is inverted from this
module's `0.0 = fully open` convention. Because the formula is expressed in
terms of the two named endpoints rather than a min/max ordering, `95 → 0.0`
and `0 → 1.0` without a special case. (The existing `servo` variant cannot
express this at all: `ServoGripper.__init__` requires `max_deg > min_deg`.)

The read-side clamp is load-bearing, not defensive habit: so-101's
`openPosition` is 95 while the servo physically travels to 100, so a raw read
of 98 maps to `-0.03` unclamped and puts an out-of-range value into the
observation vector the policy sees.

**But the clamp is bounded, and that matters.** An unbounded clamp absorbs a
mis-scaled endpoint pair as readily as it absorbs calibration slop, and the
two are not the same thing. With `open_value=95, closed_value=0` pointed at a
driver actually reporting raw units, `840 → 0.0`, `400 → 0.0`, `2 → 0.979`:
the whole upper half of that driver's travel reads "fully open" and the
policy's gripper channel freezes at a rail, silently and permanently. That is
the same failure shape as a non-finite reading, and copying the so-101 config
example onto a different gripper is a realistic way to reach it.

So `read()` clamps within a slack band and *refuses* outside it
(`_READ_SLACK = 0.25` of span in each direction). Calibration slop is small by
definition — so-101's is ~5% of span, xarm's ~1% — so the band admits every
legitimate excursion while a wrong endpoint pair becomes an error naming both
configured values. It pairs with `_preflight_gripper`, which reads before any
arm motion, so a mis-scaled pair surfaces as a startup refusal rather than a
per-tick lie.

Note the config-time message about percentage guesses saturating a raw-unit
driver fires only when the fields are *omitted*, never when they are wrong.
This band is what covers the wrong case.

Errors name what went wrong and what to do:

- missing `read_key` → `GripperRuntimeError` quoting the configured key *and*
  the keys the response actually contained.
- non-numeric value → same, quoting the value and its type.

### Wiring

Attributes: `in_state = True`, `uses_degrees = False`, `dependency_name = name`,
`arm_joint_index = None`. Everything downstream then works unchanged:

- `_build_safety` (`service.py:348`) computes `gripper_in_degrees=False`, so
  the channel gets the normalized `[0, 1]` clamp rather than a degree clamp.
- `_check_action_dim` (`service.py:359`) expects `len(state_joint_indices) + 1`.
- `_preflight_gripper` (`service.py:405`) already probes read-then-write-back
  before any arm motion, because `in_state` is true and `arm_joint_index` is
  `None`. Note the probe is **not** a no-op for this variant: with so-101's
  `open_value=95` and a raw reading of 98, read→write-back commands the jaw to
  95. Small, and precedented (`ThresholdGripper` also actuates on preflight),
  but `_preflight_gripper`'s docstring (`service.py:414-419`) currently claims
  the probe lands proportional adapters "on the same value already reported" —
  which the read clamp makes untrue here. One sentence there keeps it honest.
- `safety.joint_limits_degs` length checking (`config.py:279`) keys off
  `gripper_type == "arm_joint"`, so no trailing limit pair is expected.

Two touch-ups needed:

1. `config.py:329` — the dependency tuple `("servo", "gripper")` gains
   `"do_command"`, or the resource never resolves.
2. `gripper.py`'s `close_threshold` guard already rejects the key for any
   `kind != "gripper"`, so `do_command` inherits that rejection correctly. No
   change; noted so the next reader does not re-derive it.

## 3. `inputs` / `threshold` read fix

Independent of so-101: both variants are built on a wrong reading of the SDK.

`framesystem.InputEnabled` (`rdk@v1.0.0/robot/framesystem/framesystem.go:36-43`)
documents *"Input units are always in meters or radians"*, and the input list
length equals the kinematic model's DOF count — `fake/gripper.go:96` sizes its
slice as `make([]referenceframe.Input, len(g.model.DoF()))`. `gripper.MakeModel`
is documented as creating *"a zero DoF Model"*, which is the common case for a
gripper. So `get_current_inputs()` on a typical gripper returns an **empty
list**, and it never carries a normalized aperture.

Both `InputsGripper.read()` and `ThresholdGripper.read()` currently do
`float(values[0]) if values else 0.0` — against a 0-DOF gripper they report
`0.0` forever, silently, and the policy sees a gripper that never moves.

Fix: an actionable refusal in both readers.

```python
values = await self._gripper.get_current_inputs()
if not values:
    raise GripperRuntimeError(
        f"gripper {self.dependency_name!r} reports no kinematic DOF, so "
        "get_current_inputs() carries no aperture value; if the driver "
        'exposes a proportional DoCommand, use gripper.type="do_command"'
    )
```

Because `_preflight_gripper` reads before any arm motion, this surfaces at
startup rather than mid-episode.

The wording is conditional deliberately. For a driver that implements only
`open()`/`grab()` and has no `get`/`set` DoCommand, this refusal leaves **no**
working variant — `threshold`'s *write* path was fine, and only its read was
broken. Pointing such an operator at `do_command` unconditionally would send
them after a variant their driver cannot satisfy. Accepting that gap rather
than building a last-commanded-value fallback is a deliberate YAGNI call: no
driver we have exhibits it (so-101 and xarm both expose `get`/`set`), and a
fallback that reports the commanded value as if it were measured would feed the
policy a fiction — the same class of bug this section exists to remove.

Docs corrected in the same change, at **three** sites:

1. the module docstring in `gripper.py:12-13`, which describes `inputs` as
   "normalized 0..1" — the error this fix exists to correct;
2. README §"Gripper variants", per §6 below;
3. `safety.py:27-31`, which enumerates the normalized variants by name
   (`servo`, `gripper/inputs`, `gripper/threshold`) and needs `do_command`
   added to that list.

New wording: inputs are one value per kinematic DOF in radians or meters, so
`inputs` works only against a driver whose gripper model is *jointed*; most
gripper models, including `devrel:so101:gripper`, are zero-DOF.

Also worth recording in the README: `InputsGripper.write` catches only
`NotImplementedError`, but a Go `errors.ErrUnsupported` arrives in Python as a
`GRPCError`, so the "reconfigure to threshold" hint never fires for a Go
driver. The new empty-inputs guard makes the read fail first, which is the
path that actually runs, so this is documentation rather than a code change.

## 4. Arm `wait: False`

`service.py:660` calls `arm.move_to_joint_positions(JointPositions(values=target))`
with no `extra`. so-101's `parseWaitExtra` (`components/arm/motion.go:242-249`)
defaults to `true`, so the arm blocks until every servo settles — on every
tick. A VLA sends a fresh setpoint each tick; waiting for the previous one to
finish defeats the chunked-action design the scheduler is built around.

```python
await arm.move_to_joint_positions(
    JointPositions(values=target),
    extra={"wait": False},
)
```

No config flag. `extra` is a free-form struct and drivers ignore keys they do
not read, so this is inert on non-so-101 arms. The SDK signature accepts it:
`move_to_joint_positions(self, positions, *, extra=None, timeout=None, **kwargs)`.

Note the asymmetry with the gripper: `Gripper.do_command` has **no** `extra`
parameter, which is why the gripper's non-blocking flag has to travel inside
the command map (`write_args`) rather than beside it.

## 5. Testing

`tests/controller/test_gripper.py`:

- normalization in both directions, including the inverted so-101 bounds
  (`open=95, closed=0`) and the xarm bounds (`open=840, closed=2`)
- read clamping at both rails, specifically a raw read *above* `open_value`
- a grossly out-of-range read refused rather than clamped (a raw-units reading
  against a percent config), and a non-finite read refused rather than clamped
- `open_value` missing, `closed_value` missing, and the two equal — all
  `GripperConfigError`
- `read_key` absent from the response, and present but non-numeric
- `read_key` defaulting to `"position"`, and overridden to `"pos"`
- `write_args` merged into the emitted command, and `{}` emitting `{"set": v}` alone

`tests/fakes.py` — three changes, two of which are prerequisites rather than
additions:

- a new `FakeDoCommandGripper` recording every command it receives, with a
  configurable response key and a switch to omit the key entirely.
- **`FakeGripper.__init__` (`tests/fakes.py:143`) must be fixed first.** It does
  `self.inputs = list(inputs or [0.0])`, so `FakeGripper(inputs=[])` silently
  becomes `[0.0]` — the empty-inputs tests below are unwritable until that `or`
  becomes an explicit `if inputs is None` check.
- **`FakeArm.move_to_joint_positions` (`tests/fakes.py:37`) must record `extra`.**
  It currently accepts the kwarg and discards it, so the `wait: False` assertion
  has nothing to assert against.

`tests/controller/test_config.py`: the existing suite has a dependency test per
variant (`test_servo_gripper_adds_dependency`, `test_gripper_mode_adds_dependency`,
lines 91-100) and a `@parametrize` over the four type names
(`test_accepts_every_known_gripper_type`, line 225). Both need a `do_command`
case, plus coverage for the `write_args` validations and the missing/equal
bounds rejections.

`tests/controller/test_service.py`:

- action-dim accounting accepts `len(state_joint_indices) + 1` for the new variant
- the pre-flight probe runs for `do_command`, and its write lands *after* the
  arm move on a real tick (mirroring the existing
  `test_gripper_write_happens_after_arm_move_for_non_arm_joint_gripper`)
- `inputs` and `threshold` against a gripper returning `[]` refuse at startup,
  with no arm motion first
- the arm receives `extra={"wait": False}`

## 6. Documentation

README §"Gripper variants" gains a `do_command` entry with both worked configs:

```json
{ "type": "do_command", "name": "grip",
  "open_value": 95.0, "closed_value": 0.0,
  "write_args": { "wait": false } }
```

```json
{ "type": "do_command", "name": "grip", "read_key": "pos",
  "open_value": 840.0, "closed_value": 2.0 }
```

The trailing sentence at README:478 — "Except for `arm_joint`, every variant's
value is normalized `0.0`–`1.0` (`0` = fully open)" — is **false**, and this
change is where it gets corrected rather than extended. An earlier draft of this
spec claimed it "stays true because it describes what each *adapter* hands the
controller". That is true only for `servo` and `do_command`. `InputsGripper` and
`ThresholdGripper` both delegate to `_read_first_input`, which returns
`float(values[0])` — the driver's raw frame-system value in radians or meters,
passed straight through with no normalization on either side. That is recorded as
a known limitation in `_read_first_input`'s own docstring, and this document
contradicted it two sections later.

What the README must say instead: `servo` and `do_command` hand the controller a
normalized `0.0`–`1.0` value; `arm_joint` carries degrees per `action_units`;
`inputs`/`threshold` pass the driver's raw frame-system value through
unnormalized. Three behaviors, not one rule with one exception.

The `inputs`/`threshold` entries are otherwise rewritten per §3.

The xarm config carries a caveat: `goToPosition`
(`viam-ufactory-xarm/arm/gripper.go:307-356`) polls until the jaw settles, up
to **10 seconds**, with no `wait` escape hatch. The variant is functionally
correct there but impractical at 10 fps without an xarm-side change.

## Why `write()` validates nothing

`read()` carries five guards; `write()` carries none. That asymmetry is
deliberate and rests on an invariant both of `write()`'s callers enforce:

- The tick loop writes `float(safe[-1])` (`service.py:662`), and `safe` comes
  from `SafetyLayer.apply`, which calls `_validate` first
  (`safety.py:119` → `safety.py:74-78`). `_validate` raises `SafetyError` on any
  non-finite value anywhere in the action vector, so a diverging policy is
  refused before the gripper channel is ever computed.
- `_preflight_gripper` (`service.py:423-424`) writes back exactly what `read()`
  just returned, and `read()` has already refused non-finite and clamped into
  `[0, 1]`.

So `write()`'s input is finite and in range by construction. The `_clamp_unit`
call in it is a backstop, not a validation point — worth keeping, but not worth
an error path.

An earlier draft of this document claimed the opposite: that a `nan` action
would reach the driver as a fully-open command because `np.clip(nan, 0, 1)` is
`nan`. That was wrong — it reasoned about the clip without checking that
`_validate` runs ahead of it. Recorded here because the wrong version was
committed, and because the invariant is the reason `write()` is allowed to be
five lines with no guards.

## Out of scope

**A 1-DOF gripper model in so-101, which is what `inputs` would actually
require.** `BuildGripperModel` (`so-101/internal/geometry/gripper.go:140-181`)
emits links only — zero joints — with the jaw mesh baked at `GripperJointMin`.
Making `CurrentInputs`/`GoToInputs` meaningful means: a revolute joint in the
model, posing the mesh from the joint value instead of baking it, matching
changes in `simulated-gripper`, and accepting that the jaw becomes a planning
DOF the frame system and motion planner now see. Our side would additionally
need radian bounds to normalize against. Large blast radius for a channel a
VLA only ever needs as a scalar — `do_command` gets there without touching
kinematics.

Recorded here so the analysis is not repeated; not planned.
