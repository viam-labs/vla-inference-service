import threading

import numpy as np
import pytest

from vla.config_util import VLAError
from vla.controller.action_queue import ActionQueue, ActionQueueError


def _chunk(n, dim=2, value=0.0):
    return np.full((n, dim), value, dtype=np.float32)


def test_empty_queue_returns_none_and_zero_size():
    q = ActionQueue()
    assert q.get() is None
    assert q.qsize() == 0


def test_serves_actions_in_order():
    q = ActionQueue()
    q.merge(np.arange(6, dtype=np.float32).reshape(3, 2))
    np.testing.assert_allclose(q.get(), [0.0, 1.0])
    np.testing.assert_allclose(q.get(), [2.0, 3.0])
    assert q.qsize() == 1


def test_merge_drops_consumed_rows_and_appends_the_new_chunk():
    q = ActionQueue()
    q.merge(_chunk(3, value=1.0))
    q.get()
    q.merge(_chunk(2, value=2.0))
    assert q.qsize() == 4
    np.testing.assert_allclose(q.get(), [1.0, 1.0])  # the unconsumed tail comes first
    q.get()
    np.testing.assert_allclose(q.get(), [2.0, 2.0])


def test_clear_resets_everything():
    q = ActionQueue()
    q.merge(_chunk(3))
    q.get()
    q.clear()
    assert q.qsize() == 0
    assert q.get() is None


def test_get_returns_a_copy_not_a_view():
    q = ActionQueue()
    q.merge(_chunk(2))
    first = q.get()
    first[:] = 99.0
    q.merge(_chunk(1))  # the stored chunk must be untouched by the caller's write
    assert not np.any(q.queue == 99.0)


def test_merge_copies_the_callers_buffer():
    q = ActionQueue()
    chunk = _chunk(2)
    q.merge(chunk)
    chunk[:] = 99.0
    np.testing.assert_allclose(q.get(), [0.0, 0.0])


@pytest.mark.parametrize("bad", [[[1.0, 2.0]], np.zeros(4, dtype=np.float32), "chunk"])
def test_merge_rejects_anything_but_a_2d_ndarray(bad):
    with pytest.raises(ActionQueueError, match="2D numpy array"):
        ActionQueue().merge(bad)


def test_action_queue_error_is_a_vla_error():
    assert issubclass(ActionQueueError, VLAError)


def test_concurrent_get_never_serves_the_same_action_twice():
    q = ActionQueue()
    q.merge(np.arange(400, dtype=np.float32).reshape(200, 2))
    seen = []
    lock = threading.Lock()

    def worker():
        while (a := q.get()) is not None:
            with lock:
                seen.append(float(a[0]))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == [float(i) for i in range(0, 400, 2)]


# ---------------------------------------------------------------------------
# replace(): a landing chunk supersedes the unconsumed rows.
# ---------------------------------------------------------------------------


def _rows(values, dim=2):
    return np.array([[v] * dim for v in values], dtype=np.float32)


def test_replace_supersedes_unconsumed_rows():
    q = ActionQueue()
    q.merge(_rows([1, 2, 3, 4]))
    q.get()
    q.replace(_rows([10, 11, 12]))
    assert [q.get()[0] for _ in range(3)] == [10, 11, 12]
    assert q.get() is None


def test_replace_onto_an_empty_queue_is_a_first_merge():
    q = ActionQueue()
    q.replace(_rows([5, 6]), blend_ticks=3)
    assert [q.get()[0] for _ in range(2)] == [5, 6]


def test_replace_crossfades_the_first_blend_ticks_rows():
    q = ActionQueue()
    q.merge(_rows([0, 0, 0, 0, 0]))
    q.replace(_rows([4, 4, 4, 4, 4]), blend_ticks=3)
    # w_j = (j+1)/(b+1) with b=3: 0.25, 0.5, 0.75, then the new value.
    np.testing.assert_allclose([q.get()[0] for _ in range(5)], [1.0, 2.0, 3.0, 4.0, 4.0])


def test_replace_blend_is_bounded_by_the_old_rows_left():
    q = ActionQueue()
    q.merge(_rows([0, 0]))
    q.get()  # one old row left
    q.replace(_rows([4, 4, 4]), blend_ticks=3)
    # b = min(3, old=1, new=3) = 1: w = 1/2.
    np.testing.assert_allclose([q.get()[0] for _ in range(3)], [2.0, 4.0, 4.0])


def test_replace_blend_is_bounded_by_the_new_rows():
    q = ActionQueue()
    q.merge(_rows([0, 0, 0, 0]))
    q.replace(_rows([4]), blend_ticks=3)
    # b = min(3, old=4, new=1) = 1, and the old rows past b are discarded.
    np.testing.assert_allclose(q.get()[0], 2.0)
    assert q.get() is None


def test_replace_leaves_columns_past_blend_dims_untouched():
    q = ActionQueue()
    q.merge(np.zeros((3, 3), dtype=np.float32))
    q.replace(np.full((3, 3), 4.0, dtype=np.float32), blend_ticks=1, blend_dims=2)
    np.testing.assert_allclose(q.get(), [2.0, 2.0, 4.0])  # the gripper column takes the new value


def test_replace_refuses_a_column_mismatch():
    for blend_ticks in (0, 1):
        q = ActionQueue()
        q.merge(np.zeros((3, 2), dtype=np.float32))
        with pytest.raises(ActionQueueError, match="column"):
            q.replace(np.zeros((3, 3), dtype=np.float32), blend_ticks=blend_ticks)


def test_replace_refuses_a_non_2d_chunk():
    q = ActionQueue()
    with pytest.raises(ActionQueueError):
        q.replace(np.zeros(3, dtype=np.float32))
