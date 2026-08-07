# viam-vla-inference-service

A [Viam](https://www.viam.com) module that runs pre-trained
[LeRobot](https://github.com/huggingface/lerobot) Vision-Language-Action (VLA) policies
on a Viam machine. Primary targets are **SmolVLA** and **Evo-1**, both first-class
policies in LeRobot `main`; any other LeRobot-registered policy whose checkpoint is
self-describing (a `config.json` plus `model.safetensors` and processor files) should
load without per-policy code.

The module ships two `rdk:service:generic` resources that communicate over `DoCommand`:

- **`viam-labs:vla:policy`** — pure inference over a LeRobot checkpoint. Owns torch. No
  robot knowledge: one request carries images + joint state + a task string, the
  response carries an entire action chunk.
- **`viam-labs:vla:controller`** — the observation → inference → actuation loop. Owns
  the cameras, the arm, the gripper adapter, the action queue, and all safety clamping.
  Pure numpy; never imports torch.

Splitting them this way means the policy can be tested with no robot attached, and the
control loop can be developed and tested against a fake policy with no GPU and no torch
install at all.

## Requirements

- **Python 3.12+.** LeRobot `main` requires it; a 3.11 venv fails to resolve.
- **Install size.** A full `uv sync --extra lerobot` venv — the one that actually runs
  inference — measured **824 MB on macOS** (Apple Silicon). Breakdown: torch 356 MB,
  cmake 124 MB, cv2 119 MB, transformers 44 MB, sympy 29 MB, viam-sdk 22 MB, numpy 22 MB.
  Linux resolves `torch+cu128`/`torchvision+cu128`, which bundle CUDA runtime libraries
  the macOS wheels omit, so the Linux figure will be **substantially larger** — measure
  on the actual target image (Jetson in particular) before committing to it. The base
  install (no `lerobot` extra — config parsing, the controller's pure-numpy code, the
  wire codec) has no torch dependency at all and measured **81 MB** on the same machine.

## `#policy` — `viam-labs:vla:policy`

Has no dependencies; `validate_config` always returns `([], [])`.

### Config

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_path` | string | — | Local checkpoint directory. Also how Viam registry delivery works: viam-server interpolates `${packages.ml_model.<name>}` into a real path before this module ever sees the config. Exactly one of `model_path` / `model_hub_id` is required. |
| `model_hub_id` | string | — | A Hugging Face Hub repo id, e.g. `lerobot/smolvla_base`. Downloaded once and cached under `$VIAM_MODULE_DATA/checkpoints` (falls back to a fresh, non-persistent temp directory with a logged warning if `VIAM_MODULE_DATA` is unset, which is the normal case on a dev workstation). |
| `model_revision` | string | `"main"` | Hub revision/tag/commit SHA. Ignored for `model_path`. |
| `hf_token_env` | string | — | The **name** of an environment variable holding a Hugging Face token — never the token itself. Machine configs are readable by anyone with fleet access. |
| `device` | `auto` \| `cuda` \| `mps` \| `cpu` | `"auto"` | `auto` tries cuda → mps → cpu, first available. Covers all three deployment targets with no per-machine edits. |
| `dtype` | `auto` \| `float32` \| `bfloat16` \| `float16` | `"auto"` | Parsed and validated but **not yet applied** to loaded weights — see Limitations. |
| `warmup_inferences` | integer, 0–100 | `2` | Throwaway inferences run on synthetic input at load, before `status` reports `ready`. First-call latency on a cold GPU is often several times steady-state. |
| `load_timeout_s` | number, 0.001–86400 | `1800` | Bounds checkpoint-resolve + load + warmup as a whole. A hang (stuck download, wedged deserialize) becomes `state: failed` with a message instead of `loading` forever. |
| `rtc.enabled` | boolean | `false` | Enables the RTC processor on the policy, if the checkpoint supports it (ignored with a logged warning otherwise). |
| `rtc.execution_horizon` | integer, 1–1000 | `10` | Mirrors lerobot's `RTCConfig.execution_horizon`. |
| `rtc.prefix_attention_schedule` | `linear` \| `exp` \| `ones` \| `zeros` | `"linear"` | Mirrors `RTCConfig`. |
| `rtc.max_guidance_weight` | number, >0–1000 | `10.0` | Mirrors `RTCConfig`. |

`policy_type` is deliberately not configurable — it is read from the checkpoint's own
`config.json`. Making it a config field would only create a way to contradict the
checkpoint.

### DoCommand

| Command | Input | Output |
|---|---|---|
| `infer` | `images` (map of feature key → image payload — see below), `state` (`{"values": [f, ...]}`), `task` (string), optional `rtc` (`{"inference_delay": int, "prev_chunk_left_over": {"rows": [[...]]}}`) | `actions` (`{"rows": [[...]]}`, postprocessed, ready for the robot), `raw_actions` (same shape, policy-space — what an RTC caller feeds back as `prev_chunk_left_over`), `latency_s` |
| `specs` | — | `policy_type`, `action_dim`, `state_dim`, `n_action_steps`, `input_features`, `output_features`, `image_feature_keys`, `supports_rtc`, `rtc_enabled`, `relative_actions`, `device`, `dtype` |
| `status` | — | `state` (`idle` \| `loading` \| `ready` \| `failed`), `error` |
| `reset` | — | `{"ok": true}` — clears any cached backend state |

An image payload is `{"encoding": "jpeg"|"png"|"raw", "data": "<base64>", "height": h,
"width": w, "channels": c}` — see `src/vla/wire.py`. `infer`/`specs`/`reset` all raise
while the policy is not `ready` (still `loading`, or `failed`).

### Worked config examples

Loading happens in a **background task** — `reconfigure()` validates and returns
immediately, even for a multi-GB hub download. `infer`/`specs`/`reset` return an error
until `status` reports `ready`.

**1. Local checkpoint directory:**

```json
{
  "model_path": "/opt/viam/checkpoints/smolvla_base"
}
```

**2. Viam registry package** (`${packages.ml_model.<name>}` is substituted by
viam-server before this module sees the config — from here it's identical to a local
path):

```json
{
  "model_path": "${packages.ml_model.my-smolvla}"
}
```

**3. Hugging Face Hub:**

```json
{
  "model_hub_id": "lerobot/smolvla_base",
  "model_revision": "main",
  "hf_token_env": "HF_TOKEN",
  "device": "auto",
  "warmup_inferences": 3
}
```

## `#controller` — `viam-labs:vla:controller`

Dependencies: `policy_service`, `arm`, every resource named in `cameras`, plus the
gripper resource named in `gripper` (if any) — all computed by `validate_config` and
returned as required dependencies.

The controller tolerates a cold `policy_service`: `reconfigure()` never fails just
because the policy is still `loading` (the normal case on first boot, while a multi-GB
checkpoint downloads). It resolves lazily after the first `start`, polling `status` with
backoff for up to `policy_ready_timeout_s`.

### Config

| Field | Type | Default | Notes |
|---|---|---|---|
| `policy_service` | string | required | Name of the `viam-labs:vla:policy` dependency. |
| `arm` | string | required | Name of the Viam `arm` component to drive. |
| `cameras` | object (feature key → camera name) | required | Must cover every key the policy reports in `specs.image_feature_keys`. |
| `state_joint_indices` | array of integers | required | Maps Viam joint order (base → end-effector) onto the state-vector position the checkpoint expects. Indices, not names — Viam joint names are not guaranteed to match LeRobot feature names. |
| `gripper` | object | `{"type": "none"}` | Discriminated union — five variants, see below. |
| `task` | string | `""` | Default task instruction; overridable per `start` call. |
| `fps` | number | `10.0` | Control loop rate. |
| `mode` | `auto` \| `sequential` \| `async` \| `rtc` | `"auto"` | `rtc` is not implemented yet — see Limitations. `auto` resolves to `sequential`, unchanged from before `async` existed — `async` is explicit opt-in only, never a default, so an existing deployment's behavior never changes underneath it. See Performance for when to reach for `async`. |
| `queue_threshold` | integer | derived (`n_action_steps - 1`, once the checkpoint is known) | Consumed by `mode: "async"`: refill fires in the background once the queue has this many or fewer actions left. Unused by `sequential`. Leave unset unless you have a specific reason to override — see Performance for why the derived default is deliberately the largest value that ever makes sense. |
| `starvation_grace_ticks` | integer | `3` | Two related bounds sharing one field: (1) consecutive tick *failures* tolerated (when `safety.stop_on_error` is `false`) before the loop halts regardless; (2) under `mode: "async"`, consecutive *empty* ticks (queue drained, inference still in flight) tolerated before the loop halts unconditionally — not gated by `stop_on_error`, since an empty tick is not a failure to skip. |
| `policy_ready_timeout_s` | integer | `600` | How long, in the background, `start` waits for a cold policy before giving up. |
| `state_units` | `degrees` \| `radians` | `"degrees"` | Unit of the state vector sent to the policy. `"normalized"` is not yet supported — see Limitations. |
| `action_units` | `degrees` \| `radians` | `"degrees"` | Unit the policy's action is expressed in. |
| `image_encoding` | `jpeg` \| `png` \| `raw` | `"jpeg"` | A debugging knob (JPEG artifacts vs. the training distribution), not a tuning one. |
| `jpeg_quality` | integer, 0–100 | `90` | |
| `duration_warn_s` | number | `0.1` | Log a warning when observation assembly takes longer than this. |
| `stale_frame_warn_s` | number | `0.5` | Log a warning when a camera frame is older than this. |
| `safety.max_joint_delta_degs` | number | `8.0` | Per-tick clamp against the arm's *measured* position. Derived automatically from `max_vel_degs_per_sec` when that is set instead — see Safety. |
| `safety.max_start_delta_degs` | number | `15.0` | `start` refuses outright if the first predicted action is farther than this from the current pose. |
| `safety.max_vel_degs_per_sec` | number | — | Operator-facing velocity knob — see Safety. |
| `safety.joint_limits_degs` | array of `[min, max]` | — | One pair per action-vector dimension, **in action-vector order** (not Viam joint order) — see Safety. |
| `safety.stop_on_error` | boolean | `true` | Whether a failure producing the next action (camera read, observation assembly, the policy call) halts the loop, or is logged and the tick skipped. |

### DoCommand

| Command | Input | Output |
|---|---|---|
| `start` | optional `task`, overriding the configured default | `{"ok": true}` — returns immediately; does not wait for the policy or the loop to actually be running |
| `stop` | — | `{"ok": true}` |
| `status` | — | `state` (`idle` \| `waiting_for_policy` \| `running` \| `stopped` \| `error`), `mode`, `queue_size`, `avg_latency_s`, `measured_fps`, `clamp_counts`, `starved_ticks` (cumulative; `mode: "async"` only, see Performance), `last_error` |

### Gripper variants

A VLA emits one continuous gripper value per tick; Viam offers three different
components that can carry it, at different fidelity.

**`arm_joint`** — recommended default. The gripper rides the arm's own joint vector (SO-100-style
drivers commonly expose it as the last joint), so read and write cost no extra round trip.
Value follows `action_units` like every other joint.

```json
{ "type": "arm_joint", "joint_index": 5 }
```

**`servo`** — bidirectional via `get_position()`/`move(angle)`, both `int` degrees (1°
resolution). Value is normalized `0.0`–`1.0` and mapped onto `[min_deg, max_deg]`.

```json
{ "type": "servo", "name": "grip-servo", "min_deg": 0, "max_deg": 90 }
```

**`gripper`, `mode: "inputs"`** — the symmetric `get_current_inputs()`/`go_to_inputs()`
pair, preserving proportional control. Preferred whenever the driver implements both
(they are abstract SDK methods, so not every driver does).

```json
{ "type": "gripper", "name": "grip", "mode": "inputs" }
```

**`gripper`, `mode: "threshold"`** — read via `get_current_inputs()`, write by
thresholding the normalized value to `open()`/`grab()`. Binary fallback for drivers that
do not implement `go_to_inputs`.

```json
{ "type": "gripper", "name": "grip", "mode": "threshold", "close_threshold": 0.5 }
```

**`none`** — no gripper channel.

```json
{ "type": "none" }
```

Except for `arm_joint`, every variant's value is normalized `0.0`–`1.0` (`0` = fully
open), matching how LeRobot datasets typically encode a gripper channel.

### Full worked example

```json
{
  "policy_service": "vla-policy",
  "arm": "my-arm",
  "cameras": {
    "observation.images.top": "cam-top",
    "observation.images.wrist": "cam-wrist"
  },
  "task": "pick up the red block",
  "fps": 10.0,
  "mode": "sequential",
  "state_joint_indices": [0, 1, 2, 3, 4],
  "state_units": "degrees",
  "action_units": "degrees",
  "gripper": { "type": "arm_joint", "joint_index": 5 },
  "safety": {
    "max_start_delta_degs": 15.0,
    "max_vel_degs_per_sec": 60.0,
    "joint_limits_degs": [
      [-110, 110], [-90, 90], [-90, 90], [-90, 90], [-180, 180], [0, 90]
    ],
    "stop_on_error": true
  },
  "image_encoding": "jpeg",
  "jpeg_quality": 90
}
```

Note that `safety.max_joint_delta_degs` is **omitted**, not given some independent
number: it is derived from `max_vel_degs_per_sec / fps` (`60.0 / 10.0 = 6.0` deg/tick
here). If both are given, they must agree, or configuration fails loudly — see Safety.

## Units

**This is the section that saves the most debugging time.** Viam arms always report and
accept **degrees** (`JointPositions.values`). A LeRobot checkpoint was trained on
whatever units the recording robot used, which is frequently *not* degrees — SO-100
datasets in particular are commonly recorded in radians, or as normalized joint
positions (this module supports the first two; see Limitations for `normalized`).

`state_units` and `action_units` tell the controller which unit the checkpoint expects
on each side of inference — the arm's degrees are converted before being sent to the
policy, and the policy's output is converted back to degrees before being sent to the
arm.

**Worked example.** Suppose an SO-100 checkpoint was recorded through a driver that
reports and expects radians, but the Viam SO-100 driver reports and accepts degrees.
Add `state_units`/`action_units` to an otherwise ordinary controller config:

```json
{
  "policy_service": "vla-policy",
  "arm": "my-arm",
  "cameras": { "observation.images.top": "cam-top" },
  "state_joint_indices": [0, 1, 2, 3, 4],
  "state_units": "radians",
  "action_units": "radians"
}
```

The controller reads the arm's joint positions in degrees, converts to radians before
calling `infer`, and converts the returned action back to degrees before commanding the
arm — so a `move_to_joint_positions` call from a policy whose whole world was radians
still lands the arm at the position it actually meant.

Getting this wrong does not usually crash anything — it produces motion that is off by
a factor of ~57 (degrees vs. radians) or wildly rescaled (normalized vs. either), which
the safety layer's delta clamp will silently absorb into constant clamping. That is
exactly why `clamp_counts` exists: see Safety below.

## Safety

Applied to every action, in this fixed order, before it reaches the arm:

1. **Reject NaN/Inf** in the action vector — fail the whole chunk, never try to clamp it.
2. **Dimension check** against the current measured arm state.
3. **Per-step delta clamp**, against the arm's **currently measured** position (not the
   last commanded one — so a stalled arm cannot accumulate an ever-growing command).
4. **Joint limit clamp**, from the optional `safety.joint_limits_degs`. The list is
   indexed **in action-vector order** — one `[min, max]` pair per action dimension, in
   the same order the policy emits and `state_joint_indices` defines — with a trailing
   gripper pair only when `gripper.type == "arm_joint"`. It is **not** indexed by Viam
   joint order. Config validation enforces
   `len(joint_limits_degs) == len(state_joint_indices) + (1 if gripper.type == "arm_joint" else 0)`.
   When absent, this layer is skipped and a warning is logged once at start, naming the
   arm driver as the sole limit authority.
5. **There is no driver-side kinematic ceiling.** `move_through_joint_positions` and
   `MoveOptions` — which would carry velocity/acceleration/TCP-speed limits — ship in no
   released `viam-sdk` (installed 0.80.0, the latest on PyPI, has only
   `move_to_joint_positions`, which takes no options). The velocity bound is instead
   enforced entirely by layer 3:

   ```
   max_joint_delta_degs = max_vel_degs_per_sec / fps
   ```

   `max_vel_degs_per_sec` is the operator-facing knob (reasoning in degrees/second is
   what a human actually does); the controller derives and logs the per-tick budget this
   implies at `reconfigure()` time. If both `max_vel_degs_per_sec` and
   `max_joint_delta_degs` are configured, they must agree (within floating-point
   tolerance) or configuration fails — silently preferring one over a contradictory
   other would hide an operator mistake instead of surfacing it. Acceleration and
   TCP-speed limiting are unavailable until the SDK ships the newer call;
   `safety.max_start_delta_degs` covers the large-initial-jump case those would
   otherwise soften.
6. **Every clamp is logged and counted**, split by layer, in `status.clamp_counts`
   (`delta` / `limit` / `gripper`). **Persistent clamping is the single most likely
   sign of wrong units or wrong joint order** — it is deliberately loud rather than
   silently "handled."

A normalized gripper channel (`servo`, `gripper/inputs`, `gripper/threshold`) is exempt
from the degree-shaped delta and limit clamps — it gets its own `[0, 1]` clamp instead,
counted separately as `clamp_counts["gripper"]`.

**First-move guard.** The first action of the first chunk can be arbitrarily far from
the arm's actual current pose — a policy handed an unfamiliar initial pose can output
anything. `start` refuses outright if that first action exceeds
`safety.max_start_delta_degs`. Refusing beats slowly moving the arm somewhere nobody
asked for.

## Performance

Measured on Apple Silicon (MPS), `lerobot/smolvla_base`:

| | |
|---|---|
| `predict_chunk` latency | **~5.3 s** |
| `n_action_steps` | 50 |
| Motion a chunk buys at `fps: 10` | 5.0 s |

**Inference is slower than the motion it produces on Apple Silicon.** `mode: "sequential"`
stalls the arm for ~5.3 s between chunks — it moves roughly half the wall-clock time, in
5-second freezes.

**`mode: "async"` is the recommended mode whenever inference latency approaches or
exceeds chunk duration**, exactly the case measured here. It overlaps execution with
inference instead of stalling: the current chunk keeps running while the next one infers
in the background, merged in append mode once it lands. Duty cycle goes from ~50% (in
multi-second freezes) to ~94% (in ~0.3 s hitches at each chunk boundary) at these
measured numbers. RTC cannot do this job instead — RTC needs `delay < chunk_length` to
function at all, and here `delay` (~53 steps) exceeds `chunk_length` (50 steps), so RTC
would discard the entire chunk on every merge (permanent starvation). See
`AsyncScheduler`'s docstring in `src/vla/controller/scheduler.py` for the full
concurrency contract and the discontinuity-at-chunk-boundary tradeoff it accepts in
exchange.

That tradeoff is why `async` is explicit opt-in rather than a default: it is unambiguously
better than `sequential` whenever this latency relationship holds, but a deployment
running comfortably inside its chunk budget has no reason to accept a seam it does not
need. Other levers still apply if you want to avoid the stall/hitch tradeoff entirely:
lower `fps` (a chunk buys more wall-clock time per inference), a smaller/faster
checkpoint, or fewer denoising steps. **For anything resembling a live demo, the x86+CUDA
target remains the practical answer** — the numbers above are Mac-specific, and `async`
narrows the gap without pretending to close it as well as faster hardware would.

### Tuning `queue_threshold` — the single most actionable knob in `mode: "async"`

`queue_threshold` decides *when* the background refill fires (once the queue has this
many or fewer actions left), which decides how much runway the refill gets before the
queue actually drains. A threshold picked too low silently forfeits most of the overlap
benefit — the loop still "works," it just spends a third of its time holding position,
and without a diagnostic it looks indistinguishable from "the policy is slow." Measured
throughput delivering 150 actions at latency ≈ chunk duration × 1.06 (`n_action_steps:
50`):

| `queue_threshold` | wall time | actions/s | vs. `sequential` |
|---|---|---|---|
| — (`sequential`) | 3.20 s | 46.9 | baseline |
| 10 | 2.99 s | 50.2 | +7% |
| 30 (the old fixed default) | 2.55 s | 58.7 | +25% |
| 49 (`n_action_steps - 1`) | 2.16 s | 69.4 | +48% |

The runway a threshold buys is `queue_threshold` ticks; the runway inference *needs* is
`ceil(observed_latency × fps)` ticks. Because the queue can only ever hold at most
`n_action_steps - 1` actions before a refill must have already been requested, the
highest `queue_threshold` can ever usefully be is `n_action_steps - 1` — which is exactly
why that is the derived default (see the config table above) rather than a fixed number:
a fixed default is right for at most one checkpoint's chunk length and measurably wrong
for every other one. Override it explicitly only if you have a specific reason to trade
some of that throughput back for fresher observations at each chunk boundary (a lower
threshold fires the refill later, off a more recent — but riskier — observation).

**Two diagnostics exist so a bad threshold is never silent:**

- `AsyncScheduler` tracks its own recent inference latencies and, once it has a stable
  reading, logs a `WARNING` **once** (never every tick) if `queue_threshold <
  ceil(observed_latency × fps)` — naming the observed latency, the configured and
  required thresholds, and the consequence ("the arm will hold position for ~N tick(s)
  per chunk"), with the concrete remedy (raise `queue_threshold`, lower `fps`, or reduce
  the policy's `num_steps`).
- `status.starved_ticks` is a running total (the same shape as `clamp_counts`) of ticks
  the loop has held position because the queue was empty and a background inference was
  still in flight — distinct from the *consecutive*-run bound that escalates to a halt at
  `starvation_grace_ticks`. A deployment that never escalates can still have a
  persistently nonzero `starved_ticks`, which is exactly the silent-degradation case this
  counter exists to surface.

## Limitations

- **`mode: "rtc"` is not implemented.** `ActionQueue` supports RTC mode and is
  differentially tested against upstream, but `RTCScheduler`, two-delay bookkeeping, and
  `prev_chunk_left_over` wiring are a follow-up plan — deferred until CUDA latency is
  measured, since RTC needs `delay < chunk_length` to function at all (on the measured
  Apple Silicon numbers, `delay > chunk_length`, so it could not be validated there even
  if implemented). Configuring `mode: "rtc"` raises at `start`. `mode: "async"` (see
  Performance) is the currently-shipped answer to slow inference; it is a plain overlap,
  not RTC, and does not smooth the seam at each chunk boundary the way RTC eventually
  would.
- **`state_units`/`action_units: "normalized"` is unsupported.** It needs a source of
  per-joint min/max (dataset stats vs. explicit config) that is an open design question.
  Only `degrees` and `radians` are accepted.
- **Relative-action checkpoints are refused under RTC**, not silently mishandled — RTC's
  prefix would need re-anchoring against the cached raw state that this module does not
  yet implement, and applying guidance in the wrong coordinate frame would produce
  plausible-looking but wrong motion. Sequential mode is unaffected.
- **No driver-side velocity/acceleration ceilings.** `move_through_joint_positions` (the
  only method that consumes `MoveOptions`) ships in no released `viam-sdk`. The velocity
  bound lives entirely in the safety layer's delta clamp (see Safety); acceleration and
  TCP-speed limiting have no enforcement path at all right now.
- **`dtype` is parsed and validated but not applied.** Casting weights with
  `policy.to(dtype=...)` breaks inference on at least one target (the deserialized
  `DeviceProcessorStep` has `float_dtype=None` and keeps emitting float32 regardless).
  The checkpoint's own dtype is used as-is; a non-`"auto"` setting logs a warning rather
  than silently doing nothing. (`lerobot/smolvla_base` runs `bfloat16` natively, which is
  why this has not blocked phase-1 testing.)

## Development

```bash
mise run test        # fast suite: no torch, no network, seconds
mise run test-all     # everything, including integration/differential (needs the lerobot extra)
uv sync --extra lerobot  # required once before test-all, or before running integration/differential directly
```

- `mise run test` runs `pytest -m 'not integration and not differential'` — this is the
  suite that must stay green (and torch-free) in a base `uv sync`, with no `lerobot`
  extra installed at all. It is the payoff for the `PolicyBackend` seam: everything
  except two files (`lerobot_backend.py` and the integration tests) never imports torch.
- `mise run test-all` runs the full suite, including `@pytest.mark.integration` (a real
  `lerobot/smolvla_base` checkpoint against a fake robot — no hardware) and
  `@pytest.mark.differential` (this module's numpy `ActionQueue` port checked against
  upstream lerobot's torch implementation, on both the pinned SHA and `main`). Both need
  `uv sync --extra lerobot` first.
- `mise run build` / `mise run package` build the wheel and the deployable
  `module.tar.gz` (`meta.json` + `run.sh` + `setup.sh` + the wheel — no `docs/`, so
  internal specs and plans never leak to the module registry).
