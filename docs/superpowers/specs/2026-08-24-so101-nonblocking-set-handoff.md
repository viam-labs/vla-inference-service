# Handoff: non-blocking gripper writes in the so-101 module

Date: 2026-08-24
Target repo: `viam-devrel/so-101` (the `devrel:so101-arm` module)
Requesting repo: `viam-vla-inference-service` — see
`docs/superpowers/specs/2026-08-24-do-command-gripper-design.md`

This doc is written to be read cold by an agent session in the so-101 repo. It
assumes no context from the conversation that produced it.

## Why this change is wanted

A VLA (vision-language-action) inference module drives the SO-ARM101 in a
closed loop at 10 fps — a 100 ms budget per tick — issuing a fresh arm setpoint
and a fresh gripper setpoint every tick. It carries the policy's continuous
gripper channel onto `devrel:so101:gripper` through the gripper's DoCommand
`get`/`set` pair (`components/gripper/gripper.go:366-387`), because the typed
`CurrentInputs`/`GoToInputs` pair returns `errors.ErrUnsupported` and the arm
component drives only servos 1–5.

Every DoCommand write currently blocks for up to 2 seconds:

```go
// components/gripper/gripper.go:220-229
func (g *so101Gripper) moveToPercent(ctx context.Context, percent float64) error {
	if _, err := g.servoDo(ctx, servocmd.CmdServoMove,
		map[string]interface{}{"percent": servocmd.ClampPercent(percent)}); err != nil {
		return err
	}
	_, err := g.servoDo(ctx, servocmd.CmdServoWaitStop,
		map[string]interface{}{"timeout_ms": gripperSettleTimeoutMs})
	return err
}
```

`gripperSettleTimeoutMs` is 2000. The wait also happens while holding `g.mu`,
so a concurrent position read blocks behind it. At 10 fps this starves the
control loop: the caller is about to supersede this setpoint 100 ms from now,
so waiting for it to physically settle is wasted latency, not safety.

The arm component already solved exactly this problem for its own motion path.
`parseWaitExtra` (`components/arm/motion.go:242-249`) reads `extra["wait"]`,
defaults to `true`, and threads it into `moveJoints(..., wait bool)`, whose
settle wait is a plain `if !wait { return nil }` early return
(`components/arm/motion.go:172-175`). This change asks the gripper to follow
the pattern the arm already established.

## The change

Thread an explicit `wait bool` through `moveToPercent`, and honor a `"wait"`
key on the two DoCommand write paths. Default `true` everywhere, so nothing
that exists today changes behavior.

### 1. `moveToPercent` takes a `wait` parameter

`components/gripper/gripper.go:220`

```go
// moveToPercent commands the servo. When wait is true it blocks until the servo settles;
// a false wait returns as soon as the move is commanded, for callers issuing setpoints
// faster than the jaw can travel (a VLA control loop at 10 fps, for instance), where
// waiting for a setpoint that is about to be superseded is pure latency.
func (g *so101Gripper) moveToPercent(ctx context.Context, percent float64, wait bool) error {
	if _, err := g.servoDo(ctx, servocmd.CmdServoMove,
		map[string]interface{}{"percent": servocmd.ClampPercent(percent)}); err != nil {
		return err
	}
	if !wait {
		return nil
	}
	_, err := g.servoDo(ctx, servocmd.CmdServoWaitStop,
		map[string]interface{}{"timeout_ms": gripperSettleTimeoutMs})
	return err
}
```

### 2. A `wait` parser mirroring the arm's

New helper in `components/gripper/gripper.go`, deliberately identical in
behavior to `components/arm/motion.go:242-249`:

```go
// gripperWaitExtra resolves the settle-wait flag from a DoCommand map, defaulting to
// true so every existing caller keeps its current blocking behavior.
func gripperWaitExtra(cmd map[string]interface{}) bool {
	if cmd != nil {
		if w, ok := cmd["wait"].(bool); ok {
			return w
		}
	}
	return true
}
```

Note this reads the flag out of the **DoCommand map itself**, not out of an
`extra` argument. That is forced by the SDK: `Gripper.DoCommand` has no
`extra` parameter (unlike `Arm.MoveToJointPositions`), so a caller has no
other place to put it. Consequence: `{"set": 42.0, "wait": false}` is the
wire form.

### 3. Both DoCommand write paths honor it

`components/gripper/gripper.go:374` — the un-nested `set` path:

```go
if percentPos, ok := cmd["set"].(float64); ok {
	percentPos = servocmd.ClampPercent(percentPos)
	// ... existing lock / isMoving bookkeeping unchanged ...
	if err := g.moveToPercent(ctx, percentPos, gripperWaitExtra(cmd)); err != nil {
		return nil, err
	}
	return map[string]interface{}{"position": percentPos}, nil
}
```

`components/gripper/gripper.go:389` — the `command: "set_position"` path, same
treatment, on the `targetPercent` branch that calls `moveToPercent`. Leave the
`servo_position` (raw ticks) branch alone: it never waited in the first place.

### 4. `Open` and `Grab` keep waiting

`components/gripper/gripper.go:257` and `:274` pass `true` explicitly.

These are one-shot typed API calls whose entire contract is "the jaw is now
open" / "did I grab something". `Grab` in particular *reads the position back*
after the move to decide its boolean return
(`gripper.go:277-288` — `positionDifference > threshold`); without the settle
wait that read samples a jaw still in flight and the grab detection breaks.
This is the one place where the wait is load-bearing rather than latency.

## Back-compatibility

Everything defaults to `true`, so:

- `Open`, `Grab`, `Stop` — unchanged.
- `{"set": v}` with no `wait` key — unchanged.
- `{"command": "set_position", "percentage": v}` with no `wait` key — unchanged.
- The `simulated-gripper` model, the `devrel:so101:teleop` service, and the
  setup app all keep their current behavior without edits.

Only a caller that explicitly passes `"wait": false` sees the new path. Worth
grepping for existing `set_position` / `set` callers in-repo to confirm none of
them wants the new default — the teleop service is the most likely candidate to
*benefit* from `wait: false` at 30–50 Hz, but changing it is a separate
decision and out of scope here.

## Tests

The existing table lives in `components/gripper/gripper_test.go` and drives a
`newFakeServoArm()` whose `lastCommand(servocmd.CmdServoMove)` records what
reached the bus. `TestGripperSetPositionClampsAndCommands` (line 323) and
`TestGripperOpenCommandsServoAndWaits` (line 223) are the models to follow.

Add:

1. `{"set": 42.0, "wait": false}` issues `CmdServoMove` and **no**
   `CmdServoWaitStop`.
2. `{"set": 42.0}` (no flag) issues both — the back-compat guard.
3. `{"command": "set_position", "percentage": 42.0, "wait": false}` issues
   `CmdServoMove` and no `CmdServoWaitStop`.
4. `Open` still issues `CmdServoWaitStop` (extend or assert alongside the
   existing test at line 223).
5. `Grab` still waits, so its position read-back still classifies correctly —
   `TestGrabReportsHeldWhenTheJawStopsShortOfClosed` (line 401) should keep
   passing untouched, which is the real assertion here.
6. A non-bool `wait` (e.g. `{"set": 42.0, "wait": "false"}`) falls back to
   waiting rather than silently skipping. The type assertion in
   `gripperWaitExtra` already gives this; the test pins it, because a JSON
   string sneaking through would otherwise disable the wait on the paths that
   need it.

## Documentation

`docs/gripper.md` — document `wait` on both write commands: default `true`,
`false` returns as soon as the move is commanded, intended for callers issuing
setpoints faster than the jaw travels. Note the contrast with the arm, where
the same flag rides in `extra` rather than in the command map, and say why
(`Gripper.DoCommand` has no `extra` parameter).

## Explicitly out of scope

**Making the gripper's kinematic model 1-DOF so `CurrentInputs`/`GoToInputs`
work.** This was considered and rejected as too large for the need.

For the record, since the analysis is done: `BuildGripperModel`
(`internal/geometry/gripper.go:140-181`) emits links only — zero joints — with
the jaw mesh baked at `GripperJointMin`. `framesystem.InputEnabled`
(`rdk@v1.0.0/robot/framesystem/framesystem.go:36-43`) specifies inputs as one
value per model DOF in meters or radians, so a zero-DOF model correctly reports
an empty input list; that is why `ErrUnsupported`
(`components/gripper/gripper.go:480-486`) is not actually wrong today. Making
the pair meaningful would require a revolute joint in the model, posing the
mesh from the joint value instead of baking it, mirroring all of it in
`simulated-gripper`, and accepting the jaw as a planning DOF that the frame
system and motion planner now see. The requesting module reaches the same
outcome through DoCommand without touching kinematics.

**Anything in the arm component.** The arm's `parseWaitExtra` already does the
right thing; the calling module simply needs to pass
`extra={"wait": False}`, which is a change on its side, not here.
