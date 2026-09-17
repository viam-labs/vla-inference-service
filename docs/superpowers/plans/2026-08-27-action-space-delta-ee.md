# `action_space: "delta-ee"` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an EVO1 checkpoint trained on end-effector pose deltas, alongside the existing joint-angle path, without changing that path's behaviour in any way.

**Architecture:** One new `action_space` field on the existing `ControllerConfig`, branching in exactly five places — config parsing, the observation's state half, the safety layer, the checkpoint dimension check, and the loop's actuation call. The scheduler, action queue, pacing, starvation bounds, status reporting, camera reads, and image encoding are shared unchanged. Every conversion between a Viam pose and the dataset's 9-vector comes from a verbatim copy of the converter's own `pose.py`, never from a reimplementation.

**Tech Stack:** Python 3.12, `viam-sdk>=0.80.0`, `scipy>=1.11` (new core dependency), pytest + pytest-asyncio (`asyncio_mode = "auto"`, so async tests need no decorator).

**Spec:** `docs/superpowers/specs/2026-08-27-action-space-delta-ee-design.md`

---

## Orientation for someone new to this codebase

Read these four things before Task 1.

**What the joints path does today.** Every tick: read cameras and
`get_joint_positions()` concurrently, select `state_joint_indices`, convert
degrees → checkpoint units, send to the policy service, take one action off the
scheduler, convert back to degrees, clamp against the *measured* joints, and
write a full-width `JointPositions` with `extra={"wait": False}`. That loop is
deployed and must not change.

**What delta-EE changes.** The observation's state half comes from
`get_end_position()` instead, as a 9-vector; the action is a *relative* pose
delta composed onto the measured pose and written with `move_to_position`. Four
differences, listed in the spec's §"What is shared".

**The two silent failure modes.** Reading the state's six rotation numbers as
matrix *columns* instead of rows, and LEFT-multiplying the delta. Both yield a
valid rotation, both move the arm smoothly, neither has a runtime symptom. This
is why the conversion code is copied rather than written, and why two tests exist
whose only job is to fail if either mistake is made.

**Where the numbers come from.** `viamrobotics/xarm-open-box-eedelta`, 34,670
frames at 10 fps. Per-tick translation: median 9.31 mm, p99 28.40 mm, max
96.83 mm. Per-tick rotation: median 0.0142 rad, p99 0.0807 rad, max 0.3246 rad.
Every safety default in this plan traces to that table.

---

## File Structure

**New:**

| File | Responsibility | Tasks |
| --- | --- | --- |
| `src/vla/controller/pose.py` | Verbatim copy of the converter's pose helpers. | 1 |
| `tests/controller/test_pose.py` | Golden vectors from the converter, plus the two canaries. | 2 |

**Modified:**

| File | Responsibility | Tasks |
| --- | --- | --- |
| `pyproject.toml` | `scipy>=1.11` as a core dependency. | 1 |
| `src/vla/controller/units.py` | Gains `VectorUnits` / `to_working` / `from_working`. | 3 |
| `src/vla/controller/safety.py` | Gains `CartesianLimits` / `CartesianSafetyLayer`. | 4 |
| `src/vla/controller/config.py` | `action_space`, per-segment units, Cartesian safety knobs, the new image fit, five rejections. | 5 |
| `src/vla/controller/observation.py` | `action_space` branch on the state half; `stretch_bicubic`. | 6 |
| `src/vla/controller/service.py` | Actuation extracted per action space; pose command path; startup checks. | 7 |
| `tests/fakes.py` | `default_pose`, pose API on `FakeArm`, `PoselessArm`, `RefusingArm`. | 8 |
| `README.md` | Config table, an `action_space` section, Units, Safety. | 11 |

**Test commands:**
- One test: `uv run pytest tests/controller/test_pose.py::test_name -v`
- Fast suite: `mise run test` (`uv run pytest -m 'not integration and not differential' -v`)
- Differential: `uv sync --extra lerobot` then `uv run --no-sync pytest -m differential -v`

**Establish the baseline before Task 1.** Run `mise run test` on unmodified `main`
and record the count. Every later run is compared against it; a plan that skips
this cannot tell a regression from a pre-existing failure.

---

## Task 1: Vendor `pose.py` and add scipy

- [ ] **Step 1: Extract the source**

```bash
git -C <converter> show action-space-delta-ee:src/viam_sequence_to_lerobot/pose.py > /tmp/converter_pose.py
```

Record the commit (`git rev-parse action-space-delta-ee`). It goes in the
provenance block and in the test file's docstring.

- [ ] **Step 2: Copy it verbatim, with a provenance block**

Write `src/vla/controller/pose.py` as: a new docstring paragraph naming the repo,
branch, commit, and path; the reason for copying rather than depending (the
converter pulls `lerobot` and `pyarrow`, and the matching version is on an
unmerged branch that can be force-pushed); a statement of the two silent failure
modes; then the original docstring, then **the code byte-for-byte unchanged**.

Assert that last part programmatically while writing — compare the substring
below the closing `"""` against the source file — rather than eyeballing it.

- [ ] **Step 3: Add the dependency**

`scipy>=1.11` in `[project].dependencies`, not in the `lerobot` extra, with a
comment saying why: the controller runs the composition itself and can be
deployed on a machine that never loads a checkpoint. Then `uv sync`.

- [ ] **Step 4: Prove the copy behaves identically**

Load both files by path with `importlib.util` and assert `np.array_equal` on
`pose_state` and `state_compose` across at least three poses and three deltas.
Keep the resulting numbers — they are Task 2's goldens. Print
`abs(R - R.T).max()` for the near-vertical pose and confirm it is well above
zero; a pose where rows and columns coincide cannot guard the transpose.

---

## Task 2: `tests/controller/test_pose.py`

- [ ] **Step 1: Pin the goldens**

Three poses (near-vertical xarm, generic off-axis, identity) and three deltas
(recorded median, recorded p99, zero). Assert `pose_state`, `state_compose`, and
`orientation_vector` against the Task 1 output at `rtol=1e-12`. The docstring
must record the regeneration recipe.

- [ ] **Step 2: The inverse property**

`state_delta(s, state_compose(s, d)) == d`. The dataset was built by one and
inference runs the other; if they were not inverses, every commanded pose would
be wrong by whatever they disagreed about and no single-tick test would notice.

- [ ] **Step 3: The two canaries**

Each constructs the mistake *locally* and asserts production matches the correct
answer and **not** the mistaken one. Each also asserts the two answers are
separated by a real margin, so a future fixture that made them coincide fails
loudly instead of quietly testing nothing.

- [ ] **Step 4: Prove the canaries actually fail** — the step that makes Task 2
      worth doing.

Introduce each mistake in `src/vla/controller/pose.py`, run, record the output,
restore. Expect the transpose to fail 12+ tests and the left-multiply 6.

**Clear `__pycache__` between mutation and restore.** Two writes inside the same
second can leave the same mtime and size, and Python will reuse the stale
bytecode — which looks exactly like "the restore did not work".

- [ ] **Step 5: The quaternion-order guard**

Recompute the rotation from the SDK's own quaternion (`.i/.j/.k/.w` — the scalar
attribute is `w`, not `real`) and compare. scipy orders `(x, y, z, w)` and the SDK
`(w, i, j, k)`; a hand reorder is a plausible wrong rotation with nothing to catch
it.

---

## Task 3: Per-segment units

- [ ] **Step 1: Add the machinery**

`UnitSegment(size, unit)`, a frozen hashable `VectorUnits`, `SEGMENT_UNITS`, and
`to_working`/`from_working`. Validate at construction, not at the first tick: a
segment list that does not add up is a config mistake.

Scale factors read as "one checkpoint unit in working units" (`meters -> 1000.0`),
so `meters` is checkable by eye rather than being an inverse.

- [ ] **Step 2: Do not touch `from_degrees`/`to_degrees`**

Every joints deployment goes through them. Extend the module docstring to explain
why two APIs coexist and why the angle basis differs between them — a joint angle
is a Viam quantity, an axis-angle rotvec is not.

- [ ] **Step 3: Tests**

The mm/rad identity must be exactly `assert_array_equal`, not `allclose`:
anything else means a stray conversion crept into the common path. Plus:
independent segments, unitless never scaled, width mismatch refused rather than
broadcast (numpy would happily broadcast a 3-vector against a 6-scale array),
every unit's factor, and float32 output.

---

## Task 4: `CartesianSafetyLayer`

- [ ] **Step 1: Add it beside `SafetyLayer`**

`apply(delta) -> delta`. No `current` parameter — a delta is self-contained.
Reject non-finite and any width but 6; otherwise clamp, never reject.

- [ ] **Step 2: Scale, do not clip**

Both clamps scale the whole 3-vector by one factor. Component-wise clipping
changes the commanded *direction*. Put the reason in the class docstring; a future
reader will otherwise "simplify" it into `np.clip`.

- [ ] **Step 3: Defaults, with the table**

`max_tcp_delta_mm=40.0`, `max_tcp_rot_delta_rads=0.12`. The docstring carries the
median/p99/max table and states both halves of the choice: ~1.4× above the p99 so
in-distribution motion never clamps, and *below* the recorded maximum on purpose.

- [ ] **Step 4: Match the existing layer's conventions**

`clamp_counts` as a `Counter`, keyed `translation`/`rotation`; a warning naming
the measured magnitude and the ceiling; equality at the ceiling not counted as
clamped.

- [ ] **Step 5: Tests**

In-distribution and p99 pass untouched; the recorded maximum clamps both;
direction preserved and explicitly **not** equal to what `np.clip` would give; the
two clamps independent; zero delta never divides by zero; the argument not
mutated.

---

## Task 5: Config

- [ ] **Step 1: `ACTION_SPACES`, `JOINTS`, `DELTA_EE`, and `stretch_bicubic`**

Add to `IMAGE_FITS`. `_default_image_fit(action_space)` returns `"pad"` for joints
and `"stretch_bicubic"` for delta-EE, with the reason in its docstring: smolvla
pads inside the policy, EVO1 does not.

- [ ] **Step 2: Extract the shared velocity-derivation rule**

`SafetyConfig._resolve_max_joint_delta` becomes a module-level `_resolve_per_tick`
parameterized by field name. Same semantics, same tolerances. Both action spaces
need exactly this rule and two copies would be two chances for it to drift.

Keep the joints error text byte-compatible — existing tests match on
`max_vel_degs_per_sec` as a substring.

- [ ] **Step 3: The Cartesian half of `SafetyConfig`**

Four new fields; `parse` takes `action_space`, defaulting to `JOINTS` so direct
callers keep working. Reject the wrong action space's keys **in both directions**,
before anything else.

- [ ] **Step 4: `_parse_segment_units`**

Rejects the plain-string form with a message showing the object form. Unknown keys
rejected. State rotation is a one-choice `"unitless"` field, so a wrong value is
reported rather than ignored.

- [ ] **Step 5: `_parse_state_joint_indices`**

Required for joints, rejected for delta-EE. Extracted so `parse` does not grow a
second inline branch.

- [ ] **Step 6: Reject a non-`none` gripper under delta-EE**

Message must say *why* — no gripper channel in the action space — and point at the
out-of-band path.

- [ ] **Step 7: Tests**

Every rejection in both directions. Then
`test_an_existing_joints_config_is_unaffected_in_every_field`, spelling out each
default **by value**: a golden built from the same code would move with the bug it
is supposed to catch.

---

## Task 6: Observation

- [ ] **Step 1: `action_space` parameter**

Compare against the literal `"delta-ee"`, the way `image_fit` already compares
against literals: `config.py` owns the vocabulary and imports this module for its
warning constants, so importing it back is a cycle. Say that in a comment.

- [ ] **Step 2: Branch the arm coroutine, not the gather shape**

One arm read either way, so the latency profile is unchanged. Rename the local
from `joints` to `arm_reading` and branch the error message.

- [ ] **Step 3: `_build_pose_state`**

Call `pose_state` with `{"pose": {...}}` built from the protobuf `Pose`'s seven
fields — the shape the vendored file was written against, so it needs no adapter.
Map `AttributeError` and `ValueError` onto `ObservationError` with actionable text.

- [ ] **Step 4: `stretch_bicubic` in `_encode`**

A new `elif`, above the existing `else: raise`. Do **not** change `"stretch"` —
an existing deployment must reproduce its output byte-for-byte.

- [ ] **Step 5: Tests**

The 9-dim build from a known pose; the transpose canary at this boundary too (a
transpose applied at either layer produces the same wrong observation); no joint
read on this path; metres scaling only the position; the three refusals; and for
images, no black band under a stretch, bicubic ≠ bilinear, and a frame already at
the declared shape passing through untouched.

- [ ] **Step 6: The differential resampler test**

Compare against `_batched_resize_01` from the pinned lerobot, and include the
counter-test that BILINEAR fails the same threshold. Without the counter-test the
tolerance could loosen to accept any resampler and nobody would know.

---

## Task 7: Service

- [ ] **Step 1: Extract the joints actuation into `_command_joints`**

A pure move of the existing block out of `_loop`, with `first` becoming a
`check_start` parameter. Do this **before** adding anything, and run the suite: a
green run here proves the extraction was pure.

- [ ] **Step 2: `_command_pose`**

Convert → clamp → read the measured pose → `state_compose` → `orientation_vector`
→ `move_to_position`. Returns `False` on refusal. The docstring must explain why
this one arm-command failure is bounded rather than fatal: the joints path's
target is always inside joint space, this one additionally runs IK, and IK
declines *before* commanding motion.

- [ ] **Step 3: Branch in `_loop`, keep the tail shared**

`consecutive_rejected_moves`, bounded by the existing `starvation_grace_ticks`,
not gated by `stop_on_error` (the arm never moved). Reset on success. Pacing and
tick recording stay common.

- [ ] **Step 4: `_build_safety`, `_check_delta_ee_dims`, `_check_end_position`**

The dimension check is a straight equality — both widths are contract, not
config. `_check_end_position` runs the full decode at setup so a degenerate
orientation fails before motion, not on the first tick.

- [ ] **Step 5: `_log_cartesian_budget`**

Log the ceiling, the implied mm/s and rad/s, and one factual sentence that joint
limits on this path are enforced by the arm driver's IK. `move_to_position` takes
no speed argument, so this log is the only place an operator can read the number.

- [ ] **Step 6: Tests**

`move_to_position` used and `move_to_joint_positions` never; a zero delta
round-tripping to the measured pose exactly (the sharpest form of the whole
conversion test); ticks compounding; the clamp firing end to end with the
magnitude checked; per-segment units end to end asserting **both** what the right
conversion gives and what the wrong one would have; four pre-motion refusals; and
the three refusal-handling behaviours (survives, bounded, counter resets).

---

## Task 8: Fakes

- [ ] `default_pose()` — the near-vertical xarm regime, where a transpose is
      hardest to spot. Not the identity.
- [ ] `FakeArm.get_end_position` / `move_to_position`, snapping the reported pose
      to what was commanded, mirroring how `move_to_joint_positions` snaps
      `positions`. Without the snap, "ticks compound" is untestable.
- [ ] `PoselessArm` — raises, because real drivers raise rather than return `None`.
- [ ] `RefusingArm` — counts attempts, which is what proves the loop kept ticking.

---

## Task 9: Verify the joints path is untouched

- [ ] **Step 1: Fast suite green on the branch.** Compare the count against the
      Task 0 baseline.

- [ ] **Step 2: Run `main`'s *unmodified* tests against the new source.**

Use a **git worktree**, never the working tree:

```bash
git worktree add --detach /tmp/bcheck main
rm -rf /tmp/bcheck/src && cp -r src /tmp/bcheck/src
cp pyproject.toml uv.lock /tmp/bcheck/
```

Then run the same command on a pristine `main` worktree and confirm the two counts
are identical. That is the back-compat evidence; the in-branch test is not, since
it runs modified test files.

> Do **not** copy `main`'s `tests/` over the working tree to do this. Uncommitted
> test work is unrecoverable if you then `git checkout -- tests`.

- [ ] **Step 3: Differential suite.** `uv sync --extra lerobot`, then
      `uv run --no-sync pytest -m differential`. `--no-sync` is required or the
      pinned lerobot is silently reinstalled over anything else.

---

## Task 10: Verify EVO1 loads

- [ ] `get_policy_class("evo1")`, `Evo1Config().image_resolution`. Confirm rather
      than assume the pinned extra provides it. If it does, `src/vla/policy/`
      needs no changes: the padded 24-wide chunk is cropped by
      `Evo1ActionProcessorStep`, which `LeRobotBackend.predict_chunk` already
      routes through.

---

## Task 11: README

- [ ] `action_space` row in the controller config table, plus the four
      `safety.max_tcp_*` rows and the widened `image_fit`.
- [ ] An `### action_space` section: the state and action layouts, the three
      rejections, a worked config.
- [ ] Units section: the per-segment form and why one string cannot describe the
      vector.
- [ ] Safety section: the Cartesian clamps, the measured statistics behind the
      defaults, and one factual sentence that joint limits are the arm driver's
      job on this path.
- [ ] `image_fit`: `stretch_bicubic`, with the measured divergence numbers.

---

## Deliberately not in this plan

- **Local IK plus `move_to_joint_positions`.** Would restore `joint_limits_degs`
  and the joint delta clamp, at the cost of a kinematic model per arm. The owner
  chose `move_to_position`.
- **A gripper channel.** Not in the dataset.
- **A start-delta clamp.** A delta-EE action is already a delta; the first tick's
  magnitude is what every later tick's clamp measures.
- **`mode: "rtc"`.** Still unimplemented, for both action spaces.
