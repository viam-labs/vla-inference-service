"""Deterministic in-memory backend, so everything above it tests without torch."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .backend import PolicyBackend, PolicySpecs, resolve_image_feature_keys

_DEFAULT_CAMERA_KEYS: tuple[str, ...] = ("observation.images.top",)


class FakePolicyBackend(PolicyBackend):
    """Deterministic stand-in for a real LeRobot backend.

    `last_rtc` and `call_count` are test affordances for asserting what a
    caller passed in -- they are not part of the `PolicyBackend` contract
    and, like any plain instance attributes read from outside `_lock`, are
    only reliable when a test inspects them after all concurrent calls to
    `predict_chunk` have returned. Reading them *during* concurrent calls
    (rather than after) can observe a value from any in-flight call.
    """

    def __init__(
        self,
        *,
        action_dim: int = 6,
        n_action_steps: int = 50,
        supports_rtc: bool = True,
        relative_actions: bool = False,
        image_size: tuple[int, int] = (224, 224),
        state_dim: int | None = None,
        camera_keys: tuple[str, ...] = _DEFAULT_CAMERA_KEYS,
    ) -> None:
        self._action_dim = action_dim
        # Defaults to action_dim only because that's a legitimate value for
        # a caller who doesn't care to distinguish them -- never assume the
        # two coincide. They deliberately do not on every real checkpoint.
        self._state_dim = action_dim if state_dim is None else state_dim
        self._n_action_steps = n_action_steps
        self._supports_rtc = supports_rtc
        self._relative_actions = relative_actions
        self._image_size = image_size
        self._camera_keys = tuple(camera_keys)
        self._specs: PolicySpecs | None = None
        self._lock = threading.Lock()
        self.last_rtc: dict[str, Any] | None = None
        self.call_count = 0

    def load(
        self,
        checkpoint_dir: str,
        *,
        device: str,
        dtype: str,
        rtc: Any | None,
        unused_image_features: frozenset[str] = frozenset(),
    ) -> None:
        h, w = self._image_size
        input_features: dict[str, list[int]] = {key: [3, h, w] for key in self._camera_keys}
        input_features["observation.state"] = [self._state_dim]
        # sorted(), matching LeRobotBackend's `sorted(cfg.image_features)`
        # rather than preserving constructor order: nothing depends on the
        # order today, but a fake that orders keys differently from the real
        # backend cannot catch an ordering assumption that creeps into it.
        declared_image_keys = sorted(self._camera_keys)
        # Same validation as LeRobotBackend, via the same shared helper --
        # this is what lets a controller test exercise "checkpoint declares
        # more cameras than it consumes" with no torch/lerobot installed.
        image_keys = resolve_image_feature_keys(declared_image_keys, unused_image_features)
        self._specs = PolicySpecs(
            policy_type="fake",
            action_dim=self._action_dim,
            state_dim=self._state_dim,
            n_action_steps=self._n_action_steps,
            input_features=input_features,
            output_features={"action": [self._action_dim]},
            image_feature_keys=image_keys,
            declared_image_feature_keys=declared_image_keys,
            supports_rtc=self._supports_rtc,
            rtc_enabled=bool(rtc and getattr(rtc, "enabled", False)),
            relative_actions=self._relative_actions,
            device=device,
            dtype=dtype,
        )

    @property
    def specs(self) -> PolicySpecs | None:
        return self._specs

    def predict_chunk(self, images, state, task, rtc_kwargs):
        if self._specs is None:
            # Mirrors LeRobotBackend. Two ordering hazards -- warmup
            # running ahead of load, and a superseded reconfigure swapping
            # self._backend mid-flight -- must fail loudly here rather than
            # silently returning a chunk for a policy never loaded.
            raise RuntimeError("backend not loaded")

        # Bookkeeping only, guarded because the policy service dispatches
        # concurrent requests via asyncio.to_thread: call_count is
        # read-modify-write and last_rtc is last-writer-wins, both of which
        # race without a lock. The prediction itself stays stateless per the
        # PolicyBackend contract -- only these two test affordances share
        # mutable state across calls.
        with self._lock:
            self.call_count += 1
            # Copy rather than hold the caller's reference: a caller that
            # mutates its rtc_kwargs dict after the call returns must not
            # retroactively change what this backend recorded as received.
            self.last_rtc = None if rtc_kwargs is None else dict(rtc_kwargs)

        steps, dim = self._n_action_steps, self._action_dim
        # A fixed ramp: deterministic, and raw != processed so a controller that
        # swaps the two fails loudly instead of silently passing.
        base = np.arange(steps, dtype=np.float32).reshape(steps, 1)
        raw = np.tile(base, (1, dim)) * 0.01
        processed = raw + 1.0
        return processed, raw
