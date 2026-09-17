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
