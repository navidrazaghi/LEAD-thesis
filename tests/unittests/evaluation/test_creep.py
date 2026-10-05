"""Tests for the escape-from-stall creep.

The stall this exists for is a loop rather than a fault: zero predicted demand
means zero throttle, a stopped car sees a scene in which it is stopped, and the
same prediction comes back. Nothing in the control path restores motion. What
these check is that the escape fires only when it should, that it releases the
brake -- throttle alone moves nothing -- and above all that it is inert while
switched off, because every closed-loop number in this project was collected
with it off and must stay reproducible.

The method is exercised through a stand-in carrying only the three attributes
it touches, so the state machine is tested without building an agent, which
needs CARLA.
"""

import types

import pytest

from lead.config import LeadConfig
from lead.evaluation.agents.transfuser.transfuser_agent import TransfuserAgent


def _agent(lead_config: LeadConfig) -> types.SimpleNamespace:
    """A stand-in holding the state the real agent initialises to zero."""
    return types.SimpleNamespace(
        lead_config=lead_config,
        stationary_ticks=0,
        creep_ticks_left=0,
    )


def _creep(agent: types.SimpleNamespace, throttle: float, brake: float,
           speed: float) -> tuple[float, float]:
    """Call the real method against the stand-in."""
    return TransfuserAgent._creep_if_stuck(agent, throttle, brake, speed)


def _config(**overrides: object) -> LeadConfig:
    """A config with the creep on and whatever else the test needs."""
    lead_config = LeadConfig()
    inference = lead_config.evaluation.inference
    inference.creep_when_stuck = True
    inference.creep_after_seconds = 2.0
    inference.creep_seconds = 0.5
    inference.creep_throttle = 0.4
    inference.creep_speed_threshold = 0.1
    for name, value in overrides.items():
        setattr(inference, name, value)
    return lead_config


def _ticks_per_second(lead_config: LeadConfig) -> int:
    """The rate the creep counts in."""
    return lead_config.expert.simulation.carla_fps


class TestDisabledByDefault:
    """The switch is off, and off has to mean untouched."""

    def test_default_config_leaves_the_creep_off(self) -> None:
        """A run that does not ask for it must not get it."""
        assert LeadConfig().evaluation.inference.creep_when_stuck is False

    def test_controls_pass_through_unchanged_however_long_it_sits(self) -> None:
        """Off is not 'creeps later'; it is never."""
        lead_config = LeadConfig()
        agent = _agent(lead_config)
        for _ in range(100 * _ticks_per_second(lead_config)):
            assert _creep(agent, 0.0, 1.0, 0.0) == (0.0, 1.0)


class TestWhenItFires:
    """Only after a continuous stop, and not one tick sooner."""

    def test_nothing_happens_before_the_wait_has_elapsed(self) -> None:
        """A normal stop at a light must be left alone."""
        lead_config = _config()
        agent = _agent(lead_config)
        wait = int(2.0 * _ticks_per_second(lead_config))
        for _ in range(wait - 1):
            assert _creep(agent, 0.0, 1.0, 0.0) == (0.0, 1.0)

    def test_it_creeps_once_the_wait_has_elapsed(self) -> None:
        """The loop is broken by applying throttle the plan did not ask for."""
        lead_config = _config()
        agent = _agent(lead_config)
        wait = int(2.0 * _ticks_per_second(lead_config))
        for _ in range(wait - 1):
            _creep(agent, 0.0, 1.0, 0.0)
        throttle, brake = _creep(agent, 0.0, 1.0, 0.0)
        assert throttle == pytest.approx(lead_config.evaluation.inference.creep_throttle)

    def test_the_brake_is_released_while_creeping(self) -> None:
        """Throttle against a held brake moves nothing, so this is the point."""
        lead_config = _config()
        agent = _agent(lead_config)
        for _ in range(int(2.0 * _ticks_per_second(lead_config))):
            throttle, brake = _creep(agent, 0.0, 1.0, 0.0)
        assert brake == 0.0

    def test_moving_again_resets_the_counter(self) -> None:
        """Time spent stopped only counts if it was continuous."""
        lead_config = _config()
        agent = _agent(lead_config)
        wait = int(2.0 * _ticks_per_second(lead_config))
        for _ in range(wait - 1):
            _creep(agent, 0.0, 1.0, 0.0)
        _creep(agent, 0.5, 0.0, 5.0)
        assert agent.stationary_ticks == 0
        assert _creep(agent, 0.0, 1.0, 0.0) == (0.0, 1.0)


class TestHowLongItLasts:
    """A nudge, not a throttle held down for the rest of the route."""

    def test_the_creep_stops_after_its_configured_length(self) -> None:
        """It ends on its own even if the car has not moved."""
        lead_config = _config()
        agent = _agent(lead_config)
        rate = _ticks_per_second(lead_config)
        for _ in range(int(2.0 * rate)):
            _creep(agent, 0.0, 1.0, 0.0)
        creeping = 1
        while _creep(agent, 0.0, 1.0, 0.0)[0] != 0.0:
            creeping += 1
            assert creeping <= rate, "the creep never ended"
        assert creeping == pytest.approx(int(0.5 * rate), abs=1)

    def test_a_failed_nudge_waits_a_full_interval_before_the_next(self) -> None:
        """Otherwise the trigger is still true next tick and it never lets go."""
        lead_config = _config()
        agent = _agent(lead_config)
        rate = _ticks_per_second(lead_config)
        for _ in range(int(2.0 * rate)):
            _creep(agent, 0.0, 1.0, 0.0)
        while _creep(agent, 0.0, 1.0, 0.0)[0] != 0.0:
            pass
        # The car never moved, so the only thing that can stop a second creep
        # from starting immediately is the counter having been reset.
        assert agent.stationary_ticks < int(2.0 * rate)
        assert _creep(agent, 0.0, 1.0, 0.0) == (0.0, 1.0)


class TestSpeedThreshold:
    """What counts as stationary is a threshold, not exact zero."""

    def test_a_crawl_below_the_threshold_still_counts_as_stopped(self) -> None:
        """A model creeping a few centimetres a tick is stuck, not driving."""
        lead_config = _config()
        agent = _agent(lead_config)
        below = lead_config.evaluation.inference.creep_speed_threshold / 2.0
        for _ in range(int(2.0 * _ticks_per_second(lead_config))):
            throttle, _ = _creep(agent, 0.0, 1.0, below)
        assert throttle == pytest.approx(lead_config.evaluation.inference.creep_throttle)

    def test_a_speed_above_the_threshold_does_not(self) -> None:
        """Slow traffic is not a stall."""
        lead_config = _config()
        agent = _agent(lead_config)
        above = lead_config.evaluation.inference.creep_speed_threshold * 2.0
        for _ in range(int(10.0 * _ticks_per_second(lead_config))):
            assert _creep(agent, 0.2, 0.0, above) == (0.2, 0.0)
