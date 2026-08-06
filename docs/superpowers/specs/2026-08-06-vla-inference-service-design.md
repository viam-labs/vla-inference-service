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
  `preprocessor_config.json`, `postprocessor_config.json`.
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
delay        = ceil(latency_tracker.max() / time_per_chunk) # PREDICTED → into policy
idx_before   = queue.get_action_index()
t0 = perf_counter()
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

No pin fighting because **viam-sdk Python uses `grpclib`, not `grpcio`**, so it never touches
lerobot's quarantined `grpcio-dep` extra (which caps `grpcio<=1.73.1` / `protobuf<=6.32.0` for
`reachy2-sdk`).

Extras needed are narrow: `lerobot[smolvla]` = `transformers` + `num2words` + `accelerate`;
`lerobot[evo1]` = `transformers`.

**lerobot `main` requires Python >= 3.12.**

### Viam SDK facts

- `JointPositions.values` are **degrees** (rotational) / mm (translational), ordered spatially from
  base toward end effector.
- `MoveOptions` fields: `max_vel_degs_per_sec`, `max_acc_degs_per_sec2`,
  `max_vel_degs_per_sec_joints`, `max_acc_degs_per_sec2_joints`, `max_tcp_speed`.
- **`gripper` has no continuous write.** Its API is `open`, `grab`, `is_holding_something`, `stop`,
  `is_moving`, `get_kinematics`, `get_current_inputs`. `get_current_inputs() -> List[float]` gives a
  continuous *read*; there is no continuous setter. A VLA emits continuous gripper actions, so this
  asymmetry must be resolved in config.
- `servo` has `get_position() -> int` and `move(angle: int)` — bidirectional but **integer degrees**.
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
│              arm.MoveToJointPositions  viam-labs:vla:policy │
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
- `rtc` mirrors `RTCConfig` field-for-field. Ignored with a logged warning when
  `policy.supports_rtc()` is `False`. When RTC is enabled, the service injects the `RTCConfig` at
  load and calls `policy.init_rtc_processor()`.

### DoCommand surface

| command | input | output |
|---|---|---|
| `infer` | `images{key→b64}`, `state[]`, `task`, optional `rtc{inference_delay, prev_chunk_left_over, execution_horizon}` | `actions[][]`, `raw_actions[][]`, `latency_s` |
| `specs` | — | `policy_type`, `input_features`, `output_features`, `n_action_steps`, `action_dim`, `supports_rtc`, `rtc_enabled`, `device` |
| `status` | — | `loading` \| `ready` \| `failed`, `error` |
| `reset` | — | clears cached state |

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

  "state_joint_indices": [0, 1, 2, 3, 4],
  "state_units": "degrees",
  "action_units": "degrees",

  "gripper": { "type": "arm_joint", "joint_index": 5 },

  "safety": {
    "max_joint_delta_degs": 8.0,
    "max_start_delta_degs": 15.0,
    "max_vel_degs_per_sec": 60.0,
    "max_acc_degs_per_sec2": 120.0,
    "max_tcp_speed_mm_per_sec": 250.0,
    "stop_on_error": true
  },

  "image_encoding": "jpeg",
  "jpeg_quality": 90
}
```

**RTC enablement lives only on the policy service.** The controller has no `rtc.enabled` flag; at
startup it calls `specs` and reads `supports_rtc` / `rtc_enabled`, then selects its scheduler.
`mode` is `auto` | `sequential` | `rtc`: `auto` follows the policy, `sequential` forces the simple
path for debugging, `rtc` errors loudly if the policy cannot do it. This eliminates split-brain
misconfiguration where the two halves disagree about which algorithm is running.

**No image size in config.** The controller learns the expected resolution from
`specs.input_features` and resizes to it; a configurable size could only ever disagree with the
checkpoint. `image_encoding` (`jpeg` | `png` | `raw`) stays configurable because JPEG artifacts vs.
the training distribution is a real thing to bisect — a debugging knob, not a tuning one.

**`state_joint_indices` maps Viam joint order → state-vector position.** Viam returns the driver's
order (base→end-effector); the state vector's order comes from whatever recorded the training data.
Indices rather than names because Viam joint names are not guaranteed to match LeRobot feature
names, and an index list can be eyeballed against a dataset. `state_units` covers
`degrees` | `radians` | `normalized` — SO-100 datasets are typically not degrees, so this matters as
soon as a public checkpoint is loaded.

### Gripper block

A discriminated union, because Viam's gripper API cannot accept a continuous command:

```json
{ "type": "arm_joint", "joint_index": 5 }
{ "type": "servo", "name": "grip-servo", "min_deg": 0, "max_deg": 90 }
{ "type": "gripper", "name": "grip", "close_threshold": 0.5 }
{ "type": "none" }
```

- **`arm_joint` — recommended.** SO-100-style Viam arm drivers commonly expose the gripper as
  joint 6, so read and write both go through the arm API and the asymmetry disappears.
- `servo` — bidirectional via `get_position()` / `move(angle)`; caveat, both are `int`, so 1°
  resolution.
- `gripper` — read via `get_current_inputs()`, write by thresholding the continuous output to
  `open()` / `grab()`. Works with any Viam gripper but loses proportional control. Documented
  fallback, not the default.

### DoCommand surface

| command | input | output |
|---|---|---|
| `start` | optional `task`, `fps` | ack |
| `stop` | — | ack |
| `status` | — | `running`, `mode`, `queue_size`, `avg_latency_s`, `measured_fps`, `clamp_counts`, `last_error` |

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
  → arm.move_to_joint_positions(..., MoveOptions(...)) + gripper write
```

Joint limits come from `arm.get_kinematics()` at startup, not from config, so they cannot drift from
the actual hardware.

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
4. Joint limit clamp from `arm.get_kinematics()`.
5. `MoveOptions` velocity / acceleration / TCP caps handed to the driver.
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
| `ActionQueue` port | append continuity, RTC replace + delay trimming, delay clamping `max(0, min(delay, len))`, `get_left_over` indexing, concurrent get/merge | — |
| Differential vs upstream | port fidelity | lerobot |
| Mapping | degrees↔radians↔normalized, joint remap, 4 gripper variants, resize/encode | — |
| Safety | clamp ordering, NaN rejection, start-delta refusal | — |
| `policy` service | full DoCommand surface, RTC field plumbing, error paths, loading states | `FakePolicyBackend` |
| `controller` | loop timing, starvation, stop-on-error, reconfigure-while-running | fake arm/camera/policy |
| Integration, no robot | real `lerobot/smolvla_base`: chunk shape `[n_action_steps, action_dim]`, finite values, RTC kwargs demonstrably change output | checkpoint |
| Integration, fake arm | full loop sustains target fps | checkpoint |
| Hardware smoke | documented manual checklist, conservative limits | hardware |

Every layer above the integration line runs **without torch**, in milliseconds. That is the payoff
for the `PolicyBackend` seam even while only one real backend ships.

## Phasing

1. **Prove the plumbing.** Public `lerobot/smolvla_base` checkpoint, `infer` with synthetic images,
   assert chunk shape and finite values. No hardware. Actions may be meaningless on unfamiliar
   hardware — that is expected and not a failure.
2. **Sequential loop against a fake arm.** Full controller loop, measured fps, safety layer
   exercised.
3. **Own recordings.** Record from a Viam arm so state/action conventions are controlled end to end,
   then a real demo on hardware with conservative limits.
4. **RTC.** Enable after phase-3 latency is measured on the target device, since the delay math is
   only meaningful against real latency.

## Open questions

- **Install size is unmeasured.** Estimated 3–5 GB on Linux/CUDA. Measure before committing to
  Jetson deployment; this is the one concern no architecture choice resolves.
- **Evo-1 checkpoint availability.** SmolVLA has public checkpoints on the Hub; whether a public
  Evo-1 checkpoint exists in LeRobot format for phase 1 is unconfirmed.
- **Python 3.12+ on target devices.** lerobot `main` requires >= 3.12; confirm the Jetson image can
  provide it.
- **`normalized` state units** need per-joint min/max from somewhere. If a checkpoint requires them,
  the source (dataset stats vs. explicit config) is undecided.
