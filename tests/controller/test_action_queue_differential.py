"""Run identical operation sequences through the numpy port and upstream torch
ActionQueue, asserting they agree. Requires lerobot: `uv sync --extra lerobot`.

Aliasing scope note: every assertion below goes through `np.testing.assert_allclose`
(or an `is None` check), which compares *values* and cannot tell a copy from a
view -- two arrays with identical contents pass identically whether or not
they share memory. Copy-vs-view semantics for this port are covered
explicitly elsewhere: `tests/controller/test_action_queue.py::test_returned_actions_are_copies`
asserts that mutating a `get()` result does not corrupt the queue's stored
chunk. `test_returned_arrays_do_not_alias_internal_state` below additionally
proves aliasing behavior agrees between the two implementations themselves
(mutate what each returns, then check neither implementation's internal
state moved), rather than leaving that entirely to the port's own unit
tests.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.differential

torch = pytest.importorskip("torch")
upstream_mod = pytest.importorskip("lerobot.policies.rtc.action_queue")
rtc_config_mod = pytest.importorskip("lerobot.policies.rtc.configuration_rtc")

from vla.controller.action_queue import ActionQueue as NumpyQueue
from vla.controller.action_queue import QueueSettings


def _pair(rtc_enabled: bool):
    upstream = upstream_mod.ActionQueue(rtc_config_mod.RTCConfig(enabled=rtc_enabled))
    ours = NumpyQueue(QueueSettings(rtc_enabled=rtc_enabled))
    return upstream, ours


def _chunk(rng, n, dim):
    a = rng.standard_normal((n, dim)).astype(np.float32)
    return a, torch.from_numpy(a.copy())


def _assert_same_state(upstream, ours):
    assert upstream.qsize() == ours.qsize()
    assert upstream.empty() == ours.empty()
    assert upstream.get_action_index() == ours.get_action_index()

    u_left, o_left = upstream.get_left_over(), ours.get_left_over()
    assert (u_left is None) == (o_left is None)
    if u_left is not None:
        np.testing.assert_allclose(u_left.numpy(), o_left, rtol=1e-6)

    u_proc, o_proc = upstream.get_processed_left_over(), ours.get_processed_left_over()
    assert (u_proc is None) == (o_proc is None)
    if u_proc is not None:
        np.testing.assert_allclose(u_proc.numpy(), o_proc, rtol=1e-6)


@pytest.mark.parametrize("rtc_enabled", [False, True])
def test_randomized_operation_sequences_agree(rtc_enabled):
    rng = np.random.default_rng(1234)
    upstream, ours = _pair(rtc_enabled)

    for step in range(200):
        op = rng.integers(0, 10)
        if op < 3:
            n = int(rng.integers(1, 12))
            orig_np, orig_t = _chunk(rng, n, 4)
            proc_np, proc_t = _chunk(rng, n, 4)
            delay = int(rng.integers(-2, 15))
            idx_before = int(rng.integers(0, 5)) if rng.random() < 0.5 else None
            upstream.merge(orig_t, proc_t, delay, idx_before)
            ours.merge(orig_np, proc_np, delay, idx_before)
        elif op < 9:
            u, o = upstream.get(), ours.get()
            assert (u is None) == (o is None), f"divergence at step {step}"
            if u is not None:
                np.testing.assert_allclose(u.numpy(), o, rtol=1e-6)
        else:
            upstream.clear()
            ours.clear()
        _assert_same_state(upstream, ours)


@pytest.mark.parametrize("delay", [-5, 0, 1, 3, 999])
def test_rtc_delay_trimming_agrees(delay):
    upstream, ours = _pair(True)
    rng = np.random.default_rng(7)
    orig_np, orig_t = _chunk(rng, 6, 3)
    proc_np, proc_t = _chunk(rng, 6, 3)
    upstream.merge(orig_t, proc_t, delay)
    ours.merge(orig_np, proc_np, delay)
    _assert_same_state(upstream, ours)


def test_index_mismatch_branch_agrees():
    """The branch that logs and returns the UNCLAMPED real_delay."""
    upstream, ours = _pair(True)
    rng = np.random.default_rng(11)
    orig_np, orig_t = _chunk(rng, 8, 2)
    proc_np, proc_t = _chunk(rng, 8, 2)
    upstream.merge(orig_t, proc_t, 0)
    ours.merge(orig_np, proc_np, 0)
    for _ in range(3):
        upstream.get()
        ours.get()
    orig_np2, orig_t2 = _chunk(rng, 8, 2)
    proc_np2, proc_t2 = _chunk(rng, 8, 2)
    upstream.merge(orig_t2, proc_t2, 1, 0)  # indexes_diff=3 != real_delay=1
    ours.merge(orig_np2, proc_np2, 1, 0)
    _assert_same_state(upstream, ours)


def test_rtc_delay_trimming_agrees_with_mismatched_chunk_lengths():
    """The delay clamp is min(real_delay, len(original), len(processed)) --
    a three-term min, not two. Every case above happens to use
    equal-length original/processed chunks (matching real RTC usage where
    both come from the same policy inference call), so this exercises the
    case where the two lengths genuinely differ, catching a port that
    dropped one of the three terms.
    """
    upstream, ours = _pair(True)
    rng = np.random.default_rng(23)
    orig_np, orig_t = _chunk(rng, 8, 2)
    proc_np, proc_t = _chunk(rng, 3, 2)
    upstream.merge(orig_t, proc_t, 5)
    ours.merge(orig_np, proc_np, 5)
    _assert_same_state(upstream, ours)


def test_append_mode_with_mismatched_chunk_lengths_agrees():
    """Non-RTC append mode concatenates original_actions and
    processed_actions independently -- they don't need to be the same
    length either."""
    upstream, ours = _pair(False)
    rng = np.random.default_rng(29)
    orig_np, orig_t = _chunk(rng, 4, 2)
    proc_np, proc_t = _chunk(rng, 7, 2)
    upstream.merge(orig_t, proc_t, 0)
    ours.merge(orig_np, proc_np, 0)
    _assert_same_state(upstream, ours)

    orig_np2, orig_t2 = _chunk(rng, 3, 2)
    proc_np2, proc_t2 = _chunk(rng, 5, 2)
    upstream.merge(orig_t2, proc_t2, 0)
    ours.merge(orig_np2, proc_np2, 0)
    _assert_same_state(upstream, ours)


def test_returned_arrays_do_not_alias_internal_state():
    """assert_allclose (used everywhere above) cannot distinguish a copy
    from a view -- two arrays with equal contents compare equal whether or
    not they share a buffer. This test checks aliasing explicitly: mutate
    what `get()` and `get_processed_left_over()` hand back, then confirm
    neither implementation's subsequently-read state reflects the
    mutation, and that both implementations agree on that non-aliasing
    behavior.
    """
    upstream, ours = _pair(False)
    rng = np.random.default_rng(99)
    orig_np, orig_t = _chunk(rng, 3, 2)
    proc_np, proc_t = _chunk(rng, 3, 2)
    upstream.merge(orig_t, proc_t, 0)
    ours.merge(orig_np, proc_np, 0)

    u_action = upstream.get()
    o_action = ours.get()
    u_action[0] = -999.0
    o_action[0] = -999.0

    u_left = upstream.get_processed_left_over()
    o_left = ours.get_processed_left_over()
    assert u_left[0, 0].item() != -999.0
    assert o_left[0, 0] != -999.0
    np.testing.assert_allclose(u_left.numpy(), o_left, rtol=1e-6)

    # Mutate the leftover views too, and confirm a second read is unaffected
    # on both sides -- this is the get_left_over()/get_processed_left_over()
    # copy contract, checked against upstream rather than only in isolation.
    u_left[0, 0] = -1234.0
    o_left[0, 0] = -1234.0
    u_left2 = upstream.get_processed_left_over()
    o_left2 = ours.get_processed_left_over()
    assert u_left2[0, 0].item() != -1234.0
    assert o_left2[0, 0] != -1234.0
    np.testing.assert_allclose(u_left2.numpy(), o_left2, rtol=1e-6)
