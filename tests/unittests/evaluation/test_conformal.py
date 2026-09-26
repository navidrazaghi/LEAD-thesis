"""Tests for the caution governor's online calibrator.

The whole reason this is a calibrator rather than a threshold is the claim that
its realised risk rate converges to the target it was given, from any start and
without knowing anything about the process producing the risk. These verify that
on synthetic risk streams, where the true rate is known and the answer can be
checked rather than eyeballed.
"""

import numpy as np
import pytest

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
