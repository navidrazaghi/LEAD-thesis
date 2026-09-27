"""Tests for the caution governor's online calibrator.

The whole reason this is a calibrator rather than a threshold is the claim that
its realised risk rate converges to the target it was given, from any start and
without knowing anything about the process producing the risk. These verify that
on synthetic risk streams, where the true rate is known and the answer can be
checked rather than eyeballed.
"""

import numpy as np
import pytest

from lead.config import LeadConfig
from lead.evaluation.inference.caution import (
    surrogate_risk_event,
    target_speed_multiplier,
)
from lead.evaluation.inference.conformal import ConformalCautionCalibrator


def _drive(
    calibrator: ConformalCautionCalibrator,
    risk_of: "callable",
    num_ticks: int,
) -> list[float]:
    """Run a stream through the calibrator, returning the scalar's trajectory."""
    return [calibrator.update(risk_of(tick, calibrator.value)) for tick in range(num_ticks)]


class TestConvergence:
    """What the method promises: the realised rate tracks the target."""

    def test_constant_risk_drives_the_scalar_to_its_ceiling(self) -> None:
        """Risk on every tick is a process caution cannot fix; go maximally cautious."""
        calibrator = ConformalCautionCalibrator(target_risk=0.05, step_size=0.05)
        trajectory = _drive(calibrator, lambda tick, value: True, 500)
        assert trajectory[-1] == pytest.approx(calibrator.ceiling)

    def test_no_risk_returns_the_scalar_to_zero(self) -> None:
        """With nothing going wrong the governor must get out of the way."""
        calibrator = ConformalCautionCalibrator(
            target_risk=0.05, step_size=0.05, value=1.0,
        )
        trajectory = _drive(calibrator, lambda tick, value: False, 2000)
        assert trajectory[-1] == pytest.approx(0.0)

    @pytest.mark.parametrize("target", [0.05, 0.2, 0.5])
    def test_a_responsive_process_settles_near_the_target(self, target: float) -> None:
        """A risk that caution actually suppresses should land on the target.

        The stream here is the honest case: risk arrives with a probability
        that falls as the scalar rises, which is the relationship the governor
        assumes exists but never measures.
        """
        generator = np.random.default_rng(0)
        calibrator = ConformalCautionCalibrator(target_risk=target, step_size=0.02)

        def risk_of(tick: int, value: float) -> bool:
            del tick
            return bool(generator.random() < 0.8 * (1.0 - value))

        _drive(calibrator, risk_of, 20000)
        assert calibrator.realised_risk == pytest.approx(target, abs=0.03)

    def test_it_converges_from_either_side(self) -> None:
        generator = np.random.default_rng(1)

        def run(start: float) -> float:
            calibrator = ConformalCautionCalibrator(
                target_risk=0.1, step_size=0.02, value=start,
            )
            local = np.random.default_rng(generator.integers(1 << 30))

            def risk_of(tick: int, value: float) -> bool:
                del tick
                return bool(local.random() < 0.8 * (1.0 - value))

            _drive(calibrator, risk_of, 20000)
            return calibrator.value

        assert run(0.0) == pytest.approx(run(1.0), abs=0.1)


class TestBounds:
    """The scalar must stay inside a range a controller can act on."""

    def test_it_never_leaves_zero_to_ceiling(self) -> None:
        generator = np.random.default_rng(2)
        calibrator = ConformalCautionCalibrator(
            target_risk=0.05, step_size=0.3, ceiling=0.7,
        )
        for _ in range(5000):
            value = calibrator.update(bool(generator.random() < 0.5))
            assert 0.0 <= value <= 0.7

    def test_the_ceiling_is_respected_even_under_relentless_risk(self) -> None:
        calibrator = ConformalCautionCalibrator(
            target_risk=0.05, step_size=0.5, ceiling=0.6,
        )
        _drive(calibrator, lambda tick, value: True, 100)
        assert calibrator.value == pytest.approx(0.6)


class TestReporting:
    """A run has to be readable back without the object."""

    def test_realised_risk_counts_what_happened(self) -> None:
        calibrator = ConformalCautionCalibrator()
        for tick in range(100):
            calibrator.update(tick % 4 == 0)
        assert calibrator.num_updates == 100
        assert calibrator.num_risk_events == 25
        assert calibrator.realised_risk == pytest.approx(0.25)

    def test_realised_risk_before_any_update_is_zero_not_undefined(self) -> None:
        assert ConformalCautionCalibrator().realised_risk == 0.0

    def test_state_carries_what_the_run_log_needs(self) -> None:
        calibrator = ConformalCautionCalibrator(target_risk=0.1)
        calibrator.update(True)
        state = calibrator.state()
        assert set(state) == {
            "caution_lambda",
            "realised_risk",
            "target_risk",
            "num_updates",
        }
        assert state["target_risk"] == pytest.approx(0.1)


class TestTheLoopCloses:
    """Whether the calibrator can move the risk it adapts against.

    The convergence proofs above feed it an abstract stream. What decides
    whether any of that matters is the actuator: caution scales the target
    speed, the surrogate risk asks whether speed exceeds a threshold, and if
    the floor cannot bring speed under that threshold then risk fires on every
    tick whatever the scalar does. The scalar then pins at its ceiling, the
    realised rate stays at one, and what looks like calibration is a slow
    switch to maximum caution.

    These run the whole loop -- caution to multiplier to speed to risk to
    scalar -- and judge by the realised rate rather than by where the scalar
    ended, because a fixed point just under the ceiling is a closed loop and a
    ceiling is not.
    """

    @staticmethod
    def _drive(
        nominal_speed: float,
        caution: float,
        config: LeadConfig,
        ticks: int = 20000,
    ) -> float:
        """Realised risk rate after running the closed loop to convergence."""
        governor = config.evaluation.inference
        calibrator = ConformalCautionCalibrator(
            target_risk=governor.caution_target_risk,
            step_size=governor.caution_step_size,
            ceiling=governor.caution_ceiling,
        )
        for _ in range(ticks):
            multiplier = target_speed_multiplier(caution, calibrator.value, config)
            calibrator.update(
                surrogate_risk_event(caution, nominal_speed * multiplier, config),
            )
        return calibrator.realised_risk

    @pytest.mark.parametrize("nominal_speed", [6.0, 8.0, 10.0, 12.0, 14.0])
    def test_the_realised_rate_reaches_the_target_at_driving_speeds(
        self,
        nominal_speed: float,
    ) -> None:
        """With the shipped defaults, the loop closes rather than saturating."""
        config = LeadConfig()
        rate = self._drive(nominal_speed, 0.993, config)
        target = config.evaluation.inference.caution_target_risk
        assert rate == pytest.approx(target, abs=0.02), (nominal_speed, rate)

    def test_it_stays_off_where_nothing_is_risky(self) -> None:
        """Below the risk speed there is nothing to control and no reason to act."""
        config = LeadConfig()
        assert self._drive(4.0, 0.993, config) == pytest.approx(0.0, abs=1e-3)

    def test_no_caution_means_no_risk_at_any_speed(self) -> None:
        config = LeadConfig()
        assert self._drive(14.0, 0.0, config) == pytest.approx(0.0, abs=1e-3)

    def test_an_unreachable_threshold_saturates(self) -> None:
        """The failure the defaults were changed to avoid, kept legible.

        With a risk speed the floor cannot reach, risk fires on every tick
        however cautious the governor becomes. Nothing raises; the rate simply
        never comes down, which is why this is judged on the rate.
        """
        config = LeadConfig()
        config.evaluation.inference.caution_risk_speed_mps = 2.0
        config.evaluation.inference.caution_speed_floor = 0.4
        assert self._drive(10.0, 0.993, config) > 0.9


class TestContract:
    """A configuration that could not converge is refused at construction."""

    @pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.5])
    def test_a_target_that_is_not_a_rate_is_refused(self, target: float) -> None:
        with pytest.raises(ValueError, match="target_risk"):
            ConformalCautionCalibrator(target_risk=target)

    @pytest.mark.parametrize("step", [0.0, -0.1])
    def test_a_non_positive_step_is_refused(self, step: float) -> None:
        with pytest.raises(ValueError, match="step_size"):
            ConformalCautionCalibrator(step_size=step)

    def test_a_non_positive_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ceiling"):
            ConformalCautionCalibrator(ceiling=0.0)

    def test_a_start_outside_the_range_is_clamped_not_carried(self) -> None:
        assert ConformalCautionCalibrator(value=5.0, ceiling=1.0).value == 1.0
        assert ConformalCautionCalibrator(value=-5.0).value == 0.0


class TestWarmStart:
    """Starting from a calibrated value rather than from scratch."""

    def test_a_warm_start_is_honoured(self) -> None:
        assert ConformalCautionCalibrator(value=0.4).value == pytest.approx(0.4)

    def test_a_wrong_warm_start_is_corrected_rather_than_kept(self) -> None:
        """The calibration set informs the start; it does not commit to it."""
        generator = np.random.default_rng(3)
        calibrator = ConformalCautionCalibrator(
            target_risk=0.1, step_size=0.02, value=1.0,
        )

        def risk_of(tick: int, value: float) -> bool:
            del tick, value
            return bool(generator.random() < 0.02)

        _drive(calibrator, risk_of, 20000)
        assert calibrator.value < 0.2

    def test_the_default_start_leaves_the_governor_inert(self) -> None:
        """With no calibration run, the frozen policy drives as it always did."""
        assert ConformalCautionCalibrator().value == 0.0
