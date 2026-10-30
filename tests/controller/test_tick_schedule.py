"""TickSchedule: absolute deadlines, catch-up below a period, re-anchor above."""

import logging

import pytest

from vla.controller.service import TickSchedule


class _Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def test_schedule_anchors_at_first_nominal_not_construction():
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock)
    clock.now = 5.0  # a slow first inference before the first command
    assert s.nominal() == pytest.approx(5.0)
    assert s.reanchors == 0


def test_schedule_sleeps_to_the_absolute_deadline():
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock)
    s.nominal()
    clock.now = 0.03
    assert s.advance() == pytest.approx(0.07)
    clock.now = 0.1
    assert s.nominal() == pytest.approx(0.1)


def test_schedule_catches_up_after_a_short_overrun():
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock)
    s.nominal()
    clock.now = 0.16  # tick 0 overran its deadline (0.1) by 0.06 < one period
    assert s.advance() == 0.0
    assert s.nominal() == pytest.approx(0.1)  # still on the grid
    clock.now = 0.17
    assert s.advance() == pytest.approx(0.03)  # tick 1 is short: deadline 0.2
    assert s.reanchors == 0


def test_schedule_reanchors_after_a_long_overrun(caplog):
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock)
    s.nominal()
    clock.now = 0.35  # 0.25 past the deadline: more than one period
    with caplog.at_level(logging.WARNING, logger="vla.controller.service"):
        assert s.advance() == 0.0
    assert s.reanchors == 1
    assert s.nominal() == pytest.approx(0.35)  # the next tick starts now
    assert any("overran" in r.message for r in caplog.records)


def test_schedule_reanchors_when_a_tick_starts_more_than_a_period_late():
    # A mid-run blocking next_action (e.g. mode="sequential" at a chunk
    # boundary): the tick's slot is long gone by the time it asks for its
    # nominal time, and stamping from it would send past-due points as a burst.
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock)
    s.nominal()
    clock.now = 0.05
    s.advance()
    clock.now = 2.0  # tick 1's slot started at 0.1
    assert s.nominal() == pytest.approx(2.0)
    assert s.reanchors == 1


def test_schedule_advance_anchors_a_tick_that_never_asked_for_nominal():
    clock = _Clock(3.0)
    s = TickSchedule(0.1, clock)
    assert s.advance() == pytest.approx(0.1)
    assert s.reanchors == 0


def test_schedule_without_catch_up_restarts_the_grid_after_any_late_tick():
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock, catch_up=False)
    s.nominal()
    clock.now = 0.16
    assert s.advance() == 0.0
    assert s.nominal() == pytest.approx(0.16)
    assert s.reanchors == 0


def test_schedule_a_tick_late_by_exactly_one_period_does_not_reanchor():
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock)
    s.nominal()
    clock.now = 0.2
    assert s.advance() == 0.0
    assert s.reanchors == 0


def test_schedule_nominal_reanchor_then_advance_in_the_same_tick():
    clock = _Clock(0.0)
    s = TickSchedule(0.1, clock)
    s.nominal()
    clock.now = 0.05
    s.advance()
    clock.now = 2.0
    assert s.nominal() == pytest.approx(2.0)
    clock.now = 2.03
    assert s.advance() == pytest.approx(0.07)
    assert s.reanchors == 1


def test_restart_anchors_the_next_nominal_at_now_without_counting_an_overrun():
    now = [0.0]
    s = TickSchedule(0.1, lambda: now[0])
    s.nominal()
    s.advance()
    now[0] = 5.0
    s.restart()
    assert s.nominal() == 5.0
    assert s.reanchors == 0
