import sys
import threading

import numpy as np
import pytest

from vla.config_util import VLAError
from vla.controller.action_queue import ActionQueue, ActionQueueError, QueueSettings


def chunk(n, dim, offset=0.0):
    return np.arange(n * dim, dtype=np.float32).reshape(n, dim) + offset


# ---------------------------------------------------------------------------
# Plan's baseline suite (Task 10, Step 2).
# ---------------------------------------------------------------------------


def test_empty_queue_returns_none():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    assert q.get() is None
    assert q.empty()
    assert q.qsize() == 0


def test_append_mode_serves_actions_in_order():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(3, 2), chunk(3, 2, 100.0), real_delay=0)
    np.testing.assert_array_equal(q.get(), [100.0, 101.0])
    np.testing.assert_array_equal(q.get(), [102.0, 103.0])
    np.testing.assert_array_equal(q.get(), [104.0, 105.0])
    assert q.get() is None


def test_get_serves_processed_not_original():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(2, 2), chunk(2, 2, 50.0), real_delay=0)
    np.testing.assert_array_equal(q.get(), [50.0, 51.0])


def test_append_mode_drops_consumed_and_appends():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(3, 1), chunk(3, 1, 10.0), real_delay=0)
    q.get()
    q.merge(chunk(2, 1), chunk(2, 1, 20.0), real_delay=0)
    assert q.qsize() == 4
    assert q.get_action_index() == 0


def test_rtc_mode_replaces_and_trims_by_delay():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(5, 1), chunk(5, 1, 10.0), real_delay=2)
    assert q.qsize() == 3
    np.testing.assert_array_equal(q.get(), [12.0])


def test_rtc_delay_clamped_to_shortest_array():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(3, 1), chunk(3, 1), real_delay=99)
    assert q.qsize() == 0


def test_rtc_delay_clamp_respects_shorter_processed_array_independently():
    # The clamp is `max(0, min(real_delay, len(original_actions),
    # len(processed_actions)))` -- three terms, not two. original_actions
    # and processed_actions can differ in length (they come from
    # pre/post-processing pipelines with different horizons), and a clamp
    # that dropped the len(processed_actions) term would slice the
    # original queue too generously while still (coincidentally) looking
    # fine if the two arrays happen to be the same length -- which every
    # other test in this file uses. Here len(original)=8 > len(processed)=3,
    # and real_delay=5 sits strictly between them, so the term that binds
    # the clamp is len(processed_actions), not real_delay or len(original).
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(8, 1), chunk(3, 1, 100.0), real_delay=5)
    # clamped = min(5, 8, 3) = 3
    left_over = q.get_left_over()
    assert left_over.shape[0] == 5  # 8 - 3, not 8 - 5
    assert q.qsize() == 0  # 3 - 3, not max(3 - 5, ...)


def test_negative_delay_clamped_to_zero():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(4, 1), chunk(4, 1, 10.0), real_delay=-5)
    assert q.qsize() == 4


def test_get_left_over_returns_original_space():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(4, 1), chunk(4, 1, 100.0), real_delay=0)
    q.get()
    np.testing.assert_array_equal(q.get_left_over(), [[1.0], [2.0], [3.0]])


def test_get_processed_left_over_returns_processed_space():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(4, 1), chunk(4, 1, 100.0), real_delay=0)
    q.get()
    np.testing.assert_array_equal(q.get_processed_left_over(), [[101.0], [102.0], [103.0]])


def test_left_over_is_none_before_any_merge():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    assert q.get_left_over() is None
    assert q.get_processed_left_over() is None


def test_clear_resets_everything():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(3, 1), chunk(3, 1), real_delay=0)
    q.get()
    q.clear()
    assert q.empty()
    assert q.get_action_index() == 0
    assert q.get_left_over() is None


def test_returned_actions_are_copies():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(2, 2), chunk(2, 2, 5.0), real_delay=0)
    a = q.get()
    a[0] = 999.0
    np.testing.assert_array_equal(q.get_processed_left_over()[0], [7.0, 8.0])


def test_index_mismatch_returns_unclamped_delay(caplog):
    # Mirrors upstream _check_and_resolve_delays: warn and return real_delay
    # unchanged when the observed index delta disagrees.
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(6, 1), chunk(6, 1, 10.0), real_delay=0)
    q.get()
    q.get()
    with caplog.at_level("WARNING"):
        q.merge(
            chunk(6, 1),
            chunk(6, 1, 20.0),
            real_delay=1,
            action_index_before_inference=0,
        )
    assert "real delay" in caplog.text.lower() or "indexes diff" in caplog.text.lower()
    assert q.qsize() == 5


# ---------------------------------------------------------------------------
# Additional required work item 1: get_left_over() (policy-space) and
# get_processed_left_over() (robot-space) must be distinguishable, not both
# zeros -- and a swap of their bodies must fail a test. offset=100 makes
# original values (0..3) and processed values (100..103) unmistakably
# different, so the two tests above already catch a swap; this test adds a
# single assertion of the invariant in one place for good measure.
# ---------------------------------------------------------------------------


def test_left_over_and_processed_left_over_are_not_interchangeable():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(4, 1), chunk(4, 1, 100.0), real_delay=0)
    q.get()
    original_space = q.get_left_over()
    processed_space = q.get_processed_left_over()
    assert original_space.shape == processed_space.shape
    assert not np.array_equal(original_space, processed_space)
    np.testing.assert_array_equal(original_space, [[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(processed_space, [[101.0], [102.0], [103.0]])


# ---------------------------------------------------------------------------
# Additional required work item 3: real_delay and action_index_before_inference
# arrive as protobuf doubles in production (standing requirement 4). Accept
# the integral float form, reject a fractional one via this module's own
# exception type (standing requirement 5).
# ---------------------------------------------------------------------------


def test_merge_accepts_real_delay_as_integral_float():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(5, 1), chunk(5, 1, 10.0), real_delay=2.0)
    assert q.qsize() == 3
    np.testing.assert_array_equal(q.get(), [12.0])


def test_merge_rejects_fractional_real_delay():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    with pytest.raises(ActionQueueError, match="real_delay"):
        q.merge(chunk(5, 1), chunk(5, 1), real_delay=2.5)


def test_merge_accepts_action_index_before_inference_as_integral_float():
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(6, 1), chunk(6, 1, 10.0), real_delay=0)
    q.get()
    with pytest.raises(ActionQueueError, match="action_index_before_inference"):
        q.merge(
            chunk(6, 1),
            chunk(6, 1, 20.0),
            real_delay=1,
            action_index_before_inference=0.5,
        )
    # And the integral float form works like the plain int (no error):
    q.merge(
        chunk(6, 1),
        chunk(6, 1, 20.0),
        real_delay=1,
        action_index_before_inference=1.0,
    )


def test_action_queue_error_is_a_vla_error():
    assert issubclass(ActionQueueError, VLAError)


# ---------------------------------------------------------------------------
# Exercise the "match" branch of _check_and_resolve_delays too -- not just
# the mismatch branch above -- so a mutant that always takes one branch is
# caught regardless of which one it picks.
# ---------------------------------------------------------------------------


def test_index_match_does_not_warn(caplog):
    q = ActionQueue(QueueSettings(rtc_enabled=True))
    q.merge(chunk(6, 1), chunk(6, 1, 10.0), real_delay=0)
    q.get()  # last_index = 1
    with caplog.at_level("WARNING"):
        q.merge(
            chunk(6, 1),
            chunk(6, 1, 20.0),
            real_delay=1,
            action_index_before_inference=0,
        )
    assert caplog.text == ""
    assert q.qsize() == 5


# ---------------------------------------------------------------------------
# Additional required work item 2: strengthen the concurrency test so it
# would actually catch a removed lock.
#
# A first attempt just cranked sys.setswitchinterval() way down (1e-6s) and
# hammered get()/merge() from several threads, on the theory that a tighter
# GIL switch interval would eventually interleave the read-then-increment
# in ActionQueue.get() (`action = self.queue[self.last_index]` followed by
# `self.last_index += 1`). Verified by mutation -- temporarily replacing
# `self.lock` with a no-op contextmanager -- that attempt caught nothing:
# 0/20 trials produced a duplicate serve even with the lock removed. A
# two-statement critical section with no function-call boundary in between
# is just too fast for CPython 3.12's GIL to plausibly preempt mid-statement,
# switch interval notwithstanding.
#
# `test_concurrent_get_does_not_serve_duplicate_action` below instead widens
# the race window deterministically: it swaps the live `queue` array for a
# subclass whose `__getitem__` sleeps, without touching production code.
# Since `self.queue[self.last_index]` happens inside `get()`'s real critical
# section, the sleep (which releases the GIL) gives a concurrent thread an
# actual window to run if -- and only if -- the lock isn't excluding it.
# Verified by mutation: with the same no-op-lock swap, this technique
# produced 90 duplicate serves out of 120 across repeated trials; with the
# real Lock restored, 0/120. That is the evidence this test would catch a
# removed lock; the weaker switch-interval version below is kept only as a
# coarser smoke test for gross corruption/exceptions under mixed get()/merge()
# load, not as the mutation-verified check.
# ---------------------------------------------------------------------------


class _SlowArray(np.ndarray):
    """Test-only: sleeps inside __getitem__ to widen a race window.

    Assigning `q.queue = q.queue.view(_SlowArray)` after a merge makes the
    very next `self.queue[self.last_index]` read inside the real `get()`
    take ~5ms instead of ~0, without changing any production code path.
    """

    def __getitem__(self, idx):
        import time

        time.sleep(0.005)
        return super().__getitem__(idx)


def test_concurrent_get_does_not_serve_duplicate_action():
    q = ActionQueue(QueueSettings(rtc_enabled=False))
    q.merge(chunk(200, 1), chunk(200, 1), real_delay=0)
    q.queue = q.queue.view(_SlowArray)

    errors = []
    consumed = []
    consumed_lock = threading.Lock()

    def consume():
        try:
            for _ in range(30):
                a = q.get()
                if a is not None:
                    with consumed_lock:
                        consumed.append(float(a[0]))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=consume) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # No action value should ever be served twice: a lost lock lets two
    # consumer threads read the same last_index before either increments
    # it, so both consume the same slot.
    assert len(consumed) == len(set(consumed)), "an action was served more than once"


def test_concurrent_get_and_merge_do_not_corrupt():
    """Coarser smoke test: mixed get()/merge() load under a cranked-down
    GIL switch interval must not raise or produce a duplicate serve. Not
    mutation-verified on its own (see comment above) -- kept as a broader
    net for corruption modes (shape errors, index errors) that the
    single-purpose test above doesn't exercise.
    """
    old_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # maximize thread-scheduler interleaving
    try:
        q = ActionQueue(QueueSettings(rtc_enabled=False))
        q.merge(chunk(200, 1), chunk(200, 1), real_delay=0)

        errors = []
        consumed = []
        consumed_lock = threading.Lock()

        def consume():
            try:
                for _ in range(500):
                    a = q.get()
                    if a is not None:
                        with consumed_lock:
                            consumed.append(float(a[0]))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def produce():
            try:
                # A single producer thread guarantees globally unique offsets
                # (1000, 2000, ...) -- with multiple producers, two threads
                # would legitimately generate overlapping value ranges,
                # which would look like a duplicate-serve race but isn't
                # one.
                for i in range(50):
                    offset = 1000.0 * (i + 1)
                    q.merge(chunk(5, 1, offset), chunk(5, 1, offset), real_delay=0)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=consume) for _ in range(4)] + [
            threading.Thread(target=produce)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(consumed) == len(set(consumed)), "an action was served more than once"
    finally:
        sys.setswitchinterval(old_switch_interval)
