# VLA Inference Service — Design

**Date:** 2026-08-06
**Status:** Approved for planning

## Purpose

A Viam module that runs pre-trained Vision-Language-Action policies on a Viam machine. Primary
targets are **SmolVLA** and **Evo-1**, both first-class policies in LeRobot `main`. The module
should generalize to other LeRobot-registered policies without per-policy code.

Deployment targets, in order of use: Apple Silicon dev machine → x86 workstation with NVIDIA GPU →
Jetson-class edge device.

## Grounding

Verified against `lerobot` at `~/src/lerobot` (version `0.6.2`, `main` @ `ff7cc3de1`) and
`viam-python-sdk` at `~/src/viam-python-sdk`.

### LeRobot facts

- Both target policies are registered: `src/lerobot/policies/smolvla/` and
  `src/lerobot/policies/evo1/`. Both declare `supports_rtc() -> True`. `supports_rtc` is defined on
  the `PreTrainedPolicy` base (`policies/pretrained.py:213`).
- Generic loading path: `get_policy_class`, `make_policy`, `make_pre_post_processors` in
  `policies/factory.py`.
- A checkpoint is **self-describing**: `config.json`, `model.safetensors`,
  `policy_preprocessor.json`, `policy_postprocessor.json` (plus per-step
  `.safetensors` for normalizer stats). The processor filenames come from
  `POLICY_PREPROCESSOR_DEFAULT_NAME` / `POLICY_POSTPROCESSOR_DEFAULT_NAME` in
  `lerobot/utils/constants.py:60-61` — verified against the real
  `lerobot/smolvla_base` file listing.
  `make_pre_post_processors(cfg, pretrained_path=...)` rebuilds normalization from the checkpoint,
  so the training dataset is not needed at inference time.
- RTC (Real-Time Chunking) is exposed as kwargs on `predict_action_chunk`:
  ```python
  class ActionSelectKwargs(TypedDict, total=False):
      inference_delay: int | None
      prev_chunk_left_over: Tensor | None
      execution_horizon: int | None
  ```
- `select_action` **hard-asserts RTC is off**: `"RTC is not supported for select_action, use it with
  predict_action_chunk"` (`modeling_smolvla.py:254`). RTC therefore requires
  `predict_action_chunk`.
- `_rtc_enabled()` reads `config.rtc_config.enabled`. `RTCConfig` fields: `enabled=True`,
  `prefix_attention_schedule=LINEAR`, `max_guidance_weight=10.0`, `execution_horizon=10`,
  `debug=False`, `debug_maxlen=100`.
- `policy.init_rtc_processor()` re-wires the RTC processor onto an already-constructed model, so RTC
  can be enabled after load on a checkpoint whose `rtc_config` is `None`.
- Upstream inference strategies live behind an ABC in `rollout/inference/`: `base.py`
  (`InferenceEngine`), `sync.py`, `rtc.py`, `factory.py`. **`RTCInferenceEngine` requires a
  `robot_wrapper: ThreadSafeRobot`** — it reads robot state from its own background thread — so it
  cannot be driven from a request/response service.
- `policies/rtc/action_queue.py::ActionQueue` holds **two parallel arrays**: `original_queue`
  (policy space, feeds `prev_chunk_left_over`) and `queue` (postprocessed, executed by the robot).
  Its only torch usage is `.clone()`, `torch.cat`, and slicing — no autograd, no devices.
- `lerobot.async_inference.PolicyServer` (gRPC) transports **pickled `TimedObservation` objects**,
  so any client needs lerobot installed. It cannot serve as a decoupling boundary.

The upstream RTC control loop (`rollout/inference/rtc.py`):

```python
time_per_chunk = 1.0 / fps
prev_actions = queue.get_left_over()                       # policy-space leftover
latency      = latency_tracker.max()
delay        = ceil(latency / time_per_chunk) if latency else 0   # PREDICTED → into policy
idx_before   = queue.get_action_index()
t0 = perf_counter()
if prev_actions is not None:                                # zero-pad / truncate to horizon
    prev_actions = _normalize_prev_actions_length(prev_actions, rtc_config.execution_horizon)
chunk = predict_action_chunk(obs, inference_delay=delay, prev_chunk_left_over=prev_actions)
new_delay = ceil((perf_counter() - t0) / time_per_chunk)    # MEASURED → into merge
queue.merge(original, processed, new_delay, idx_before)
```

Two distinct delays: a **predicted** delay telling RTC how much of the prefix to trust, and a
**measured** delay used to trim the returned chunk on merge.

### Dependency resolution (measured, not assumed)

`viam-sdk` + local `lerobot[smolvla,evo1]` resolve cleanly: **89 packages, zero conflicts.**

| package | resolved |
|---|---|
| viam-sdk | 0.80.0 |
| torch | 2.11.0 (`+cu128` on linux) |
| transformers | 5.5.4 |
| protobuf | 6.33.5 |
| numpy | 2.2.6 |

No pin fighting because **viam-sdk Python uses `grpclib`, not `grpcio`**, so it never pulls
lerobot's `grpcio-dep` extra (`grpcio>=1.73.1,<2.0.0`, `protobuf>=6.31.1,<8.0.0`) nor the separate
`reachy2` extra, which is where the restrictive `grpcio<=1.73.1` / `protobuf<=6.32.0` caps actually
live.

Extras needed are narrow: `lerobot[smolvla]` = `transformers` + `num2words` + `accelerate`;
`lerobot[evo1]` = `transformers`.

**lerobot `main` requires Python >= 3.12.**

### Viam SDK facts

- `JointPositions.values` are **degrees** (rotational) / mm (translational), ordered spatially from
  base toward end effector.
- **`move_through_joint_positions` is UNRELEASED — corrected 2026-08-07.** The claims below were
  read from the local dev checkout at `/Users/nick.hehr/src/viam-python-sdk`, which is ahead of
  every published release. Verified against the *installed* package: **viam-sdk 0.80.0, the latest
  on PyPI, exposes only `get_joint_positions` and `move_to_joint_positions` on `Arm`.**
  `MoveOptions` exists as a generated proto type but **no method consumes it**. The feature landed
  in the SDK repo as `9ec2f6c86` / `d2d766d0b` and has not shipped.

  Consequence: **driver-side kinematic ceilings are unavailable.** `MoveOptions` velocity,
  acceleration, and TCP-speed limits cannot be applied on any pinned SDK version.

  **Resolution — enforce the velocity bound ourselves.** Config takes
  `max_vel_degs_per_sec`, and the controller derives the per-step delta clamp from it:

  ```
  max_joint_delta_degs = max_vel_degs_per_sec / fps
  ```

  This is arguably the better design regardless. It gives one knob instead of two overlapping
  ones, the bound is enforced by code this project tests rather than by a driver whose compliance
  varies, and it removes a dependency on an unreleased API. What is genuinely lost is
  acceleration limiting and TCP-speed capping, neither of which the delta clamp can express —
  those return if and when the SDK ships the call. `check_start` already covers the large-initial-
  jump case that acceleration limits would otherwise soften.

- `move_to_joint_positions(positions, *, extra, timeout)` (the only available write) takes no
  options, and `MoveToJointPositionsRequest` carries only `name` / `positions` / `extra`.
- `MoveOptions` fields: `max_vel_degs_per_sec`, `max_acc_degs_per_sec2`,
  `max_vel_degs_per_sec_joints`, `max_acc_degs_per_sec2_joints`, and `max_tcp_speed` — the last in
  **meters per second**, not mm/s.
- Every scalar `MoveOptions` field has **explicit presence**: an unset field reads back as `0.0`,
  indistinguishable from an explicit zero. Unconfigured limits must be *omitted*, not zero-filled,
  or the arm is told not to move.
- `gripper` supports continuous values in both directions: `get_current_inputs() -> List[float]`
  reads them and `go_to_inputs(values: List[float])` (`gripper.py:217`, client at `client.py:144`,
  documented as `GoToInputs`) writes them. The rest of the API is `open`, `grab`,
  `is_holding_something`, `stop`, `is_moving`, `get_kinematics`. Both input methods are abstract, so
  a given driver may not implement them.
- `servo` has `get_position() -> int` and `move(angle: int)` — bidirectional but **integer degrees**.
- **`get_kinematics()` returns `KinematicsReturn`** — `Tuple[KinematicsFileFormat, bytes]` or
  `Tuple[KinematicsFileFormat, bytes, Mapping[str, Mesh]]`: raw URDF or SVA file bytes, or
  `UNSPECIFIED`. The Python SDK explicitly "cannot yet parse a kinematics model" and states that
  "implementations are responsible for their own limit checking" (`arm.py:161-165`). Reading joint
  limits from it would mean owning a URDF *and* SVA parser plus an unparseable-file fallback.
- `${packages.ml_model.<name>}` resolves to the package download directory and works in any module
  config string field. Registry delivery therefore needs no dedicated config field.

## Approach

**Single venv, `PolicyBackend` seam, lerobot pinned to a git SHA.** Module core depends on
`viam-sdk`; `lerobot[smolvla,evo1]` is pinned to a SHA in the same venv and imported **lazily**
inside `LeRobotBackend` so config validation and module startup work even when the backend cannot
load.

Of the three stated motivations for avoiding a lerobot dependency:

- **Dependency conflicts with viam-sdk** — does not hold empirically (see above). Dropped as a
  design driver.
- **Backend lock-in** — real; addressed by the `PolicyBackend` interface seam.
- **Install size / cold start** — the real remaining cost, almost entirely torch + CUDA libs
  (estimated 3–5 GB on Linux/CUDA, ~1 GB on Mac; **not yet measured**). No architecture fixes this;
  only a different runtime does.

### Rejected alternatives

**Sidecar subprocess with a custom protocol.** Core stays viam-sdk-only; backend runs in a separate
venv over a Unix socket with a msgpack protocol. Buys hard isolation, crash containment, and
near-free remote-GPU support — but costs protocol design, process lifecycle (spawn/health/restart/
backpressure), and two venvs in the tarball before anything works. Since the conflict concern
evaporated, this solves a problem we do not have. Revisit only if the venv proves unshippable on
Jetson.

**Vendoring a minimal lerobot subset.** lerobot's policy modules import broadly across
`lerobot.processor`, `lerobot.configs`, `lerobot.constants`, `lerobot.utils`; the "minimal" subset
would not stay minimal. It also makes tracking upstream *harder*, and forfeits the path where
`evo1` and future policies arrive for free.

**Adopting lerobot's `InferenceEngine` / `Robot` ABCs** (implementing a `ViamRobot(Robot)` adapter
so `create_inference_engine` drives the loop). This would deliver sync + RTC maintained upstream for
roughly nine adapter methods. Rejected in favor of architectural independence: it would hand the
control loop to lerobot, push safety clamps inside `send_action()`, and couple the module to two
churning ABCs instead of one comparatively stable call (`predict_action_chunk`). The accepted cost
is owning the `ActionQueue` port and the scheduler — mitigated by the differential test below.

**`rdk:service:mlmodel` for the inference resource.** `Infer` offers typed binary tensors and
`Metadata()` is a natural home for feature specs. Rejected because Viam's tensor types have no
string type, so the task instruction would be smuggled through as a uint8 byte array, and DoCommand
would still be needed for reset/specs/task switching. The efficiency argument does not survive the
numbers: VLA inputs are 224×224, so two cameras is ~15 KB as JPEG, ~20 KB base64 — 200 KB/s at
10 Hz over a local socket.

## Architecture

Two resources in one Python module, both `rdk:service:generic`:

```
viam-labs:vla:policy      pure inference. No robot knowledge.
viam-labs:vla:controller  deps, control loop, scheduler. No torch.
```

```
┌─ viam-server ──────────────────────────────────────────────┐
│  camera(s) ──┐                                             │
│  arm ────────┼──► viam-labs:vla:controller                 │
│  gripper ────┘         │  loop @ fps:                      │
│                        │   1. sample obs                   │
│                        │   2. infer ───────┐               │
│                        │   3. execute      │               │
│                        ▼                   ▼               │
│         arm.MoveThroughJointPositions  viam-labs:vla:policy │
└────────────────────────────────────────────┼───────────────┘
                                             ▼
                                    PolicyBackend (ABC)
                                             │
                                    LeRobotBackend  (lazy import)
```

**Inference is pure.** One request carries images + state + task; the response carries the entire
action chunk. The action queue and timing live in the controller. Consequences:

- No hidden per-caller state in the policy service; two callers cannot corrupt each other.
- `predict_action_chunk` is used rather than `select_action` — which is what RTC *requires*.
- The only code that touches the arm is also the only code that reasons about timing, so safety
  logic stays in one file.

**Separate controller** so the loop can be developed and tested against a fake policy, and the
policy tested with no robot attached.

### Division of labor

| Upstream (lerobot) | This module |
|---|---|
| `predict_action_chunk` | `ActionQueue` numpy port |
| `RTCProcessor.denoise_step`, `get_prefix_weights` | Scheduler + delay bookkeeping |
| pre/post-processor pipelines | Viam observation/action mapping |
| policy config + weight loading | Safety layer |

The prefix-guidance math stays upstream. What is ported is bookkeeping.

## Component: `viam-labs:vla:policy`

No dependencies. `validate()` returns `([], [])`.

### Config

```json
{
  "model_path": "${packages.ml_model.my-smolvla}",
  "model_hub_id": "lerobot/smolvla_base",
  "model_revision": "main",
  "hf_token_env": "HF_TOKEN",

  "device": "auto",
  "dtype": "auto",
  "warmup_inferences": 2,

  "rtc": {
    "enabled": true,
    "execution_horizon": 10,
    "prefix_attention_schedule": "linear",
    "max_guidance_weight": 10.0
  }
}
```

- **Exactly one** of `model_path` / `model_hub_id` is required; `validate()` errors on zero or both.
  Three delivery mechanisms, two fields: local path, registry (via `${packages...}` interpolation),
  hub. Hub downloads cache under `$VIAM_MODULE_DATA`.
- `policy_type` is deliberately absent — read from the checkpoint's `config.json`. Making it
  configurable only creates a way to contradict the checkpoint.
- `device: "auto"` → `cuda` → `mps` → `cpu`, first available. Covers all three deployment targets
  with no per-machine config edits.
- `hf_token_env` names an **environment variable** holding the Hugging Face token, never the token
  itself — machine configs are readable by anyone with fleet access.
- `dtype` is `auto` | `float32` | `bfloat16` | `float16`. `auto` selects `bfloat16` on CUDA and
  `float32` elsewhere, since MPS `bfloat16` support is uneven.
- `warmup_inferences` runs N throwaway inferences on synthetic input at load, before `status` flips
  to `ready`. First-call latency on a cold GPU is often several times steady-state, and RTC's delay
  tracker would otherwise calibrate against that outlier.
- `rtc` mirrors `RTCConfig` field-for-field. Ignored with a logged warning when
  `policy.supports_rtc()` is `False`. When RTC is enabled, the service injects the `RTCConfig` at
  load and calls `policy.init_rtc_processor()`.

### DoCommand surface

| command | input | output |
|---|---|---|
| `infer` | `images{key→b64}`, `state[]`, `task`, optional `rtc{inference_delay, prev_chunk_left_over}` | `actions[][]`, `raw_actions[][]`, `latency_s` |
| `specs` | — | `policy_type`, `input_features`, `output_features`, `n_action_steps`, `action_dim`, `supports_rtc`, `rtc_enabled`, `relative_actions`, `device` |
| `status` | — | `loading` \| `ready` \| `failed`, `error` |
| `reset` | — | clears cached state |

`execution_horizon` is deliberately **not** in the per-request `rtc` block. Upstream passes only
`inference_delay` and `prev_chunk_left_over` to `predict_action_chunk`
(`rollout/inference/rtc.py:326`), and `denoise_step` falls back to
`self.rtc_config.execution_horizon` when the argument is `None`. Keeping it config-only means one
source of truth, consistent with RTC enablement living on the policy service.

**`LeRobotBackend.predict_chunk` owns prefix normalization.** Immediately before calling
`predict_action_chunk`, it pads with zeros or truncates the decoded `prev_chunk_left_over` to its
configured `execution_horizon`, matching `_normalize_prev_actions_length`
(`rollout/inference/rtc.py:83, 320-323`). This is not optional bookkeeping: `modeling_rtc.py:189-190`
shrinks `execution_horizon` to the leftover's length when the leftover is shorter, which silently
changes `get_prefix_weights` and therefore the guidance applied. Because `execution_horizon` is
config-only, the controller *cannot* perform this step — it does not know the value — so the
backend must, and the controller sends the raw leftover at whatever length its queue holds. The
normalization itself is a pure numpy helper the backend calls, so it stays testable without torch.

**Relative-action checkpoints are refused, not silently mishandled.** When a preprocessor contains
an enabled `RelativeActionsProcessorStep`, upstream re-anchors the prefix against the cached raw
state via `reanchor_relative_rtc_prefix`, using the **processed** leftover
(`queue.get_processed_left_over()`) rather than the policy-space one, *before* length normalization
(`rollout/inference/rtc.py:159-176, 305-319`). This design's leftover path is absolute-action only.
So the backend detects the step at load, surfaces `relative_actions` in `specs`, and `mode: rtc`
**errors on such a checkpoint** rather than sending an un-rebased prefix — which would apply
guidance in the wrong coordinate frame and produce plausible-looking but wrong motion. Sequential
mode is unaffected. Implementing re-anchoring later is tractable because the `ActionQueue` port
already carries `get_processed_left_over()`.

**The response carries both `actions` and `raw_actions`** because `ActionQueue` maintains two
parallel arrays: `original_queue` in policy space (feeds `prev_chunk_left_over`) and `queue` in
postprocessed space (executed by the robot). The controller needs both. Confusing them is the most
likely RTC bug, so it is explicit in the wire format rather than implicit.

### `PolicyBackend` ABC

```
load(checkpoint_dir, device, dtype, rtc_config) -> None
predict_chunk(images, state, task, rtc_kwargs) -> (actions, raw_actions)
specs -> dict
reset() -> None
```

`LeRobotBackend` implements it with lazy `lerobot` imports. `FakePolicyBackend` returns
deterministic chunks, enabling the whole DoCommand surface to be tested without torch.

### Loading

Loading happens in a **background task**; `reconfigure` validates and returns immediately. A
multi-GB hub download inside `reconfigure()` would stall the module's reconfigure loop and can trip
viam-server timeouts. `infer` and `specs` return a "model loading" error until ready. A failed load
leaves the resource alive and diagnosable rather than crash-looping.

## Component: `viam-labs:vla:controller`

Dependencies: `policy_service`, `arm`, all `cameras` values (required); gripper resource (optional,
per gripper block).

### Config

```json
{
  "policy_service": "vla-policy",
  "arm": "my-arm",
  "cameras": {
    "observation.images.top":   "cam-top",
    "observation.images.wrist": "cam-wrist"
  },

  "task": "pick up the red block",
  "fps": 10.0,
  "mode": "auto",
  "queue_threshold": 30,
  "starvation_grace_ticks": 3,
  "policy_ready_timeout_s": 600,

  "state_joint_indices": [0, 1, 2, 3, 4],
  "state_units": "degrees",
  "action_units": "degrees",

  "gripper": { "type": "arm_joint", "joint_index": 5 },

  "safety": {
    "max_joint_delta_degs": 8.0,
    "max_start_delta_degs": 15.0,
    "max_vel_degs_per_sec": 60.0,
    "max_acc_degs_per_sec2": 120.0,
    "max_tcp_speed_m_per_sec": 0.25,
    "joint_limits_degs": [[-110, 110], [-90, 90], [-90, 90], [-90, 90], [-180, 180], [0, 90]],
    "stop_on_error": true
  },

  "image_encoding": "jpeg",
  "jpeg_quality": 90
}
```

**The controller must tolerate a cold policy.** It needs `specs` for two things — scheduler
selection and the image resize target — but the policy answers with a "model loading" error until a
possibly multi-GB download completes, which is the *normal* case on first boot. So controller
`reconfigure()` never fails on a loading policy: it stores config, returns, and fetches `specs`
lazily after the first `start`, polling `status` with backoff up to `policy_ready_timeout_s`
(default 600) in the background while reporting `waiting_for_policy`. Making `reconfigure` depend on the
machine in a config-error state every cold boot.

**RTC enablement lives only on the policy service.** The controller has no `rtc.enabled` flag; on
its first `start` it reads `supports_rtc` / `rtc_enabled` from `specs`, then selects its scheduler.
`mode` is `auto` | `sequential` | `rtc`: `auto` follows the policy, `sequential` forces the simple
path for debugging, `rtc` errors loudly if the policy cannot do it. This eliminates split-brain
misconfiguration where the two halves disagree about which algorithm is running.

**No image size in config.** The controller learns the expected resolution from
`specs.input_features` and resizes to it; a configurable size could only ever disagree with the
checkpoint. `image_encoding` (`jpeg` | `png` | `raw`) stays configurable because JPEG artifacts vs.
the training distribution is a real thing to bisect — a debugging knob, not a tuning one. `raw` is
base64 of the packed `uint8` HWC buffer, carried alongside an explicit `{height, width, channels}`
per image so the decoder never infers shape; it exists for fidelity comparisons, not production.

**`state_joint_indices` maps Viam joint order → state-vector position.** Viam returns the driver's
order (base→end-effector); the state vector's order comes from whatever recorded the training data.
Indices rather than names because Viam joint names are not guaranteed to match LeRobot feature
names, and an index list can be eyeballed against a dataset. `state_units` covers
`degrees` | `radians` | `normalized` — SO-100 datasets are typically not degrees, so this matters as
soon as a public checkpoint is loaded.

### Gripper block

A discriminated union, because a VLA emits one continuous gripper value while Viam offers three
different components that could carry it, with different fidelity:

```json
{ "type": "arm_joint", "joint_index": 5 }
{ "type": "servo",   "name": "grip-servo", "min_deg": 0, "max_deg": 90 }
{ "type": "gripper", "name": "grip", "mode": "inputs" }
{ "type": "gripper", "name": "grip", "mode": "threshold", "close_threshold": 0.5 }
{ "type": "none" }
```

- **`arm_joint` — recommended default.** SO-100-style Viam arm drivers commonly expose the gripper
  as joint 6, so read and write both ride the arm API and the gripper value stays in the same
  vector as the joints — no second round trip per tick.
- `servo` — bidirectional via `get_position()` / `move(angle)`; caveat, both are `int`, so 1°
  resolution.
- `gripper` with `mode: "inputs"` — the symmetric pair `get_current_inputs()` /
  `go_to_inputs(values)`, preserving proportional control. Preferred over `threshold` whenever the
  driver implements them; both are abstract methods, so not every driver will.
- `gripper` with `mode: "threshold"` — read via `get_current_inputs()`, write by thresholding the
  continuous action to `open()` / `grab()`. Loses proportional control. The fallback for drivers
  that do not implement `go_to_inputs`. `close_threshold` applies only in this mode.

`validate()` rejects `close_threshold` outside `mode: "threshold"` rather than silently ignoring it.

**Gripper value convention.** For `arm_joint` the value is a joint angle and follows `action_units`
like every other joint. For `gripper/inputs` and `gripper/threshold` it is normalized `0.0`–`1.0`
(0 = fully open), matching how LeRobot datasets typically encode a gripper channel; `servo` maps
that same normalized range onto `min_deg`–`max_deg`. `close_threshold` is compared against the
normalized value.

**The degrees-based safety clamps skip the normalized gripper channel.** `max_joint_delta_degs` and
`joint_limits_degs` are meaningful for `arm_joint` (where the gripper *is* a joint in degrees) but
not for `servo`, `gripper/inputs`, or `gripper/threshold`, where the channel is `0.0`–`1.0`. In
those modes the gripper dimension is clamped to `[0, 1]` instead, and `joint_limits_degs` carries no
trailing gripper pair. Leaving degree limits nominally "applied" to a 0–1 channel would mean they
never fire — a limit that silently does nothing is worse than one that is explicitly absent.

### DoCommand surface

| command | input | output |
|---|---|---|
| `start` | optional `task`, `fps` | ack, immediately |
| `stop` | — | ack |
| `status` | — | `state`, `mode`, `queue_size`, `avg_latency_s`, `measured_fps`, `clamp_counts`, `last_error` |

`start` **acks immediately** rather than blocking on the policy. Waiting out
`policy_ready_timeout_s` (default 600 s) inline would exceed the deadline most DoCommand callers
use, so the "still loading" error would never reach anyone. Instead `start` transitions to
`waiting_for_policy` and the timeout, if it expires, lands in `last_error` where `status` can report
it.

`state` is `idle` | `waiting_for_policy` | `running` | `stopped` | `error`. `mode` reports the
*configured* value until `specs` resolves it, then the resolved one — `status` labels which.

### Scheduler

`ChunkScheduler`, pure numpy:

- **`ActionQueue`** — mechanical numpy port of the upstream class, **same method names** so upstream
  diffs stay reviewable. `.clone()` → `.copy()`, `torch.cat` → `np.concatenate`.
- **`SequentialScheduler`** — append mode; blocking `infer` when the queue drains. Phase 1.
- **`RTCScheduler`** — background task; when `qsize() <= queue_threshold`, fire `infer` with the
  predicted delay; merge on return with the measured delay. Delay math exactly as upstream.

### Data flow

Observation assembly, per tick:

```python
t0 = time.perf_counter()
*frames, joints = await asyncio.gather(
    *[cams[k].get_image() for k in camera_keys],
    arm.get_joint_positions(),
)
```

Gathered, not sequential — at 10 Hz the total budget is 100 ms and two serial camera reads can
consume most of it. The batch is timestamped at `t0`; if assembly exceeds `1/fps`, log a
stale-observation warning, because RTC's delay math assumes a fresh observation and degrades
silently otherwise.

```
ViamImage → decode → resize to specs.input_features[key] → encode → b64

JointPositions.values (degrees, base→EE)
  → select by state_joint_indices
  → append gripper value (per gripper block)
  → convert degrees → state_units
```

Action return path:

```
response: actions[] (postprocessed) + raw_actions[] (policy space)
  → scheduler.merge(raw, processed, measured_delay, idx_before)
  → queue.get() per tick → one action vector
  → split: arm joints | gripper value
  → convert action_units → degrees
  → safety clamp
  → arm.move_through_joint_positions([JointPositions(...)], options=MoveOptions(...))
    + gripper write (per gripper block)
```

**The per-tick write is `move_through_joint_positions` with a single waypoint, not
`move_to_joint_positions`.** The latter accepts no `MoveOptions`, so it cannot carry the velocity,
acceleration, or TCP-speed ceilings — using it would silently drop every kinematic limit in the
safety config. The single-element list is the cost of keeping those ceilings on the hot path.

The `MoveOptions` builder sets only the fields present in config. Unset scalars read back as `0.0`,
which an arm driver cannot distinguish from an explicit zero — zero-filling would command the arm
to hold still.

## Error handling

| failure | response |
|---|---|
| Inference raises (OOM, shape, CUDA) | policy returns error; controller applies `stop_on_error` → stop arm, halt loop, surface `last_error` |
| Latency > chunk duration, sequential mode | loop runs slower than `fps`, silently changing the policy's timebase — warn loudly; abort after N consecutive |
| `measured_delay >= chunk_length`, RTC mode | every action is trimmed away and the queue starves permanently — detect explicitly and stop |
| Queue starvation | hold current position for `starvation_grace_ticks`, then stop the arm. **Never extrapolate** |
| Camera read fails | fail the whole tick. Do **not** substitute a black frame or reuse a stale one — both silently corrupt policy input in ways that look like bad model behavior |
| Arm command fails | stop immediately |
| `close()` | cancel loop, stop arm, join inference thread |
| Reconfigure while running | stop loop, stop arm, rebuild, **do not auto-resume** — require explicit `start` |

## Safety

Applied in order:

1. Reject NaN/inf in the action vector — fail the chunk rather than clamping it.
2. Dimension check against `specs.action_dim`.
3. Per-step delta clamp against the **current measured** position, not the last commanded one — so
   a stalled arm cannot accumulate an ever-growing command.
4. Joint limit clamp from the **optional** `safety.joint_limits_degs` config. **The list is indexed
   in action-vector order** — one `[min, max]` pair per action dimension, in the same order the
   policy emits and `state_joint_indices` defines, with a trailing gripper pair **only** when
   `gripper.type == "arm_joint"` (the sole mode whose gripper channel is in degrees). It is *not*
   indexed by Viam joint order. `validate()` enforces
   `len(joint_limits_degs) == len(state_joint_indices) + (1 if gripper.type == "arm_joint" else 0)`
   and fails configuration otherwise, because a silent off-by-one here clamps the wrong joint. That
   rule is a proxy: `action_dim` is unknowable at validate time, so once `specs` resolves, the
   controller additionally cross-checks the total against `specs.action_dim` and refuses to `start`
   on a mismatch. If the checkpoint emits a gripper channel while `gripper.type == "none"`, that is
   an error rather than a silent trailing-dimension drop. The clamp runs
   after `action_units → degrees` and before `JointPositions` is built. When absent, this
   layer is skipped and a warning is logged once at start, naming the arm as the sole limit
   authority. Limits are *not* read from `arm.get_kinematics()`: that returns raw URDF or SVA bytes,
   and the Python SDK cannot parse a kinematics model, so honoring it would mean owning two parsers
   plus an unparseable-file fallback — scope disproportionate to a prototype whose delta clamp and
   `MoveOptions` ceilings already bound motion. The tradeoff is that config limits can drift from
   hardware; the arm driver remains the backstop.
5. ~~`MoveOptions` ceilings handed to the driver.~~ **Removed — the API is unreleased** (see the
   Viam SDK facts above). The velocity bound is instead enforced at layer 3: config takes
   `max_vel_degs_per_sec` and the controller derives `max_joint_delta_degs = max_vel_degs_per_sec
   / fps`. Log the derived value at startup so an operator can see the per-tick budget their
   velocity limit implies. Acceleration and TCP-speed limiting are unavailable until the SDK
   ships `move_through_joint_positions`; `check_start` covers the large-initial-jump case.
6. **Log whenever a clamp engages,** and expose `clamp_counts` in `status`. Persistent clamping is
   the signature of wrong units or wrong joint order — the primary diagnostic, so it must be loud
   rather than silently correct.

**First-move guard.** The first action of the first chunk can be arbitrarily far from the arm's
current pose, and a policy handed an unfamiliar initial pose can output anything. `start` refuses
outright if the first predicted action exceeds `max_start_delta_degs`. Refusing beats moving slowly
to a place nobody asked for.

## Testing

**The differential test is the centerpiece.** Run identical operation sequences through upstream
`ActionQueue` (torch) and the numpy port, asserting equality. This converts a hand-port into a
mechanically checked property. A CI job runs it against lerobot **main** as well as the pinned SHA,
so upstream changing merge semantics breaks the build instead of the robot.

| layer | covers | needs |
|---|---|---|
| `ActionQueue` port | append continuity, RTC replace + delay trimming, delay clamping `max(0, min(real_delay, len(original), len(processed)))`, the `_check_and_resolve_delays` branch that logs and returns the **unclamped** `real_delay` when `last_index - action_index_before_inference != real_delay`, `get_left_over` indexing, concurrent get/merge | — |
| Prefix normalization | zero-pad and truncate paths of `_normalize_prev_actions_length`, including the shorter-than-horizon case that would otherwise shrink `execution_horizon` | — |
| Differential vs upstream | port fidelity | lerobot |
| Mapping | degrees↔radians↔normalized, joint remap, 5 gripper variants, resize/encode | — |
| Safety | clamp ordering, NaN rejection, start-delta refusal, skipped limit layer when `joint_limits_degs` absent, `MoveOptions` omits unconfigured fields rather than zero-filling | — |
| `policy` service | full DoCommand surface, RTC field plumbing, error paths, loading states | `FakePolicyBackend` |
| `controller` | loop timing, starvation, stop-on-error, reconfigure-while-running, cold-policy `start` → `waiting_for_policy` → `running`, `action_dim` mismatch refusal, `mode: rtc` refusing a `relative_actions` checkpoint | fake arm/camera/policy |
| Integration, no robot | real `lerobot/smolvla_base`: chunk shape `[n_action_steps, action_dim]`, finite values, RTC kwargs demonstrably change output | checkpoint |
| Integration, fake arm | full loop sustains target fps | checkpoint |
| Hardware smoke | documented manual checklist, conservative limits | hardware |

Every layer above the integration line runs **without torch**, in milliseconds. That is the payoff
for the `PolicyBackend` seam even while only one real backend ships.

## Measured: inference latency (2026-08-07, phase 1)

First real numbers, `lerobot/smolvla_base` on Apple Silicon via MPS:

| | |
|---|---|
| `predict_chunk` latency | **~5.3 s** (10 Euler denoising steps) |
| `n_action_steps` | 50 |
| Chunk duration at `fps: 10` | 5.0 s of motion |
| Load time | ~26 s cold (pulls the SmolVLM2-500M backbone), ~6 s warm |
| Checkpoint dtype | **bfloat16** natively — weights are not float32 |
| Image features | `observation.images.camera1/2/3`, each `[3, 256, 256]` |
| `action_dim` / `state_dim` | 6 / 6 |

**The Mac dev target cannot close the loop at 10 Hz.** Inference (5.3 s) slightly
exceeds the motion a chunk buys (5.0 s), so:

- **Sequential mode** stalls the arm for 5.3 s between chunks — it moves roughly
  half the wall time, in 5-second freezes. Usable to prove plumbing, not to
  demo behavior.
- **RTC would not rescue it either.** `measured_delay ≈ ceil(5.3 / 0.1) = 53`
  steps against a 50-step chunk, which is precisely the
  `measured_delay >= chunk_length` starvation condition in the error table
  above: every action gets trimmed away and the queue never fills.

This does not change the architecture, but it does constrain the phasing: the
hardware demo needs the x86+CUDA target, not the Mac. Three levers if a Mac demo
is ever wanted — lower `fps` so a chunk buys more wall-clock time, reduce the
policy's denoising steps (quality tradeoff), or use a smaller checkpoint.

The bfloat16 finding retroactively justifies not casting weights: the model
already runs mixed precision correctly (float32 inputs, bf16 parameters) with no
dtype-mismatch error on MPS. A `policy.to(dtype=...)` cast would have broken a
configuration that works by default.

## Phasing

1. **Prove the plumbing.** Public `lerobot/smolvla_base` checkpoint, `infer` with synthetic images,
   assert chunk shape and finite values. No hardware. Actions may be meaningless on unfamiliar
   hardware — that is expected and not a failure.
2. **Sequential loop against a fake arm.** Full controller loop, measured fps, safety layer
   exercised.
3. **Own recordings.** Record from a Viam arm so state/action conventions are controlled end to end,
   then a real demo on hardware with conservative limits. **Use the x86+CUDA target** — the
   measured MPS latency above rules out a closed-loop Mac demo at 10 Hz.
4. **RTC.** Enable after phase-3 latency is measured on the target device, since the delay math is
   only meaningful against real latency. Relative-action checkpoints stay refused until
   prefix re-anchoring is implemented.

## Open questions

- **Install size — measured on macOS, still unmeasured on Linux/CUDA.** A full
  `uv sync --extra lerobot` venv on Apple Silicon is **824 MB**, well under the 1 GB estimate.
  Breakdown: torch 356 MB, cmake 124 MB, cv2 119 MB, transformers 44 MB, sympy 29 MB,
  viam-sdk 22 MB, numpy 22 MB.

  Linux resolves `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from
  `download.pytorch.org/whl/cu128`, which bundle CUDA runtime libraries the macOS wheels omit,
  so the Linux figure will be several times larger. Measure on the actual Jetson image before
  committing to that target — it remains the one concern no architecture choice resolves.

  One easy win if size becomes binding: **cmake is 124 MB of pure runtime waste.** lerobot
  declares it in core dependencies only because `opencv-python-headless` needs it *to build* on
  some platforms; nothing imports it at inference time. Excluding it, and swapping
  `opencv-python-headless` for a lighter decode path if lerobot's usage allows, would cut roughly
  30% of the non-torch footprint.
- **Evo-1 checkpoint availability.** SmolVLA has public checkpoints on the Hub; whether a public
  Evo-1 checkpoint exists in LeRobot format for phase 1 is unconfirmed.
- **Python 3.12+ on target devices.** lerobot `main` requires >= 3.12; confirm the Jetson image can
  provide it.
- **`normalized` state units** need per-joint min/max from somewhere. If a checkpoint requires them,
  the source (dataset stats vs. explicit config) is undecided.
