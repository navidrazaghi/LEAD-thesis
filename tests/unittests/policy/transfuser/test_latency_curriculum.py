"""Tests for the latency curriculum's label re-anchoring.

The property that matters is the one the whole transform exists for: a plan
shifted to its execution tick must carry no systematic lag. Constant velocity is
the case where that is checkable in closed form, because a straight trajectory
at constant speed looks identical from every point along it -- so the shifted
label must equal the unshifted one exactly, and any offset is the artefact.
"""

import numpy as np
import pytest

from lead.policy.transfuser.utils.latency_curriculum import (
    sample_shift_index,
    shifted_planning_label,
)

# Eight waypoints at 4 Hz over two seconds, as the default config produces.
_HORIZON = 8
_STRIDE_S = 0.25


def _straight_line(
    speed_mps: float,
    num_ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Future poses of an ego driving straight ahead at a constant speed."""
    distances = speed_mps * _STRIDE_S * np.arange(1, num_ticks + 1, dtype=np.float64)
    waypoints = np.stack([distances, np.zeros_like(distances)], axis=1)
    return waypoints, np.zeros(num_ticks, dtype=np.float64)


def _arc(radius_m: float, speed_mps: float, num_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """Future poses of an ego on a constant-radius turn."""
    travelled = speed_mps * _STRIDE_S * np.arange(1, num_ticks + 1, dtype=np.float64)
    angle = travelled / radius_m
    waypoints = np.stack(
        [radius_m * np.sin(angle), radius_m * (1.0 - np.cos(angle))],
        axis=1,
    )
    return waypoints, angle


class TestConstantVelocityHasNoSystematicLag:
    """The headline property: shifting the label must not displace the plan."""

    @pytest.mark.parametrize("shift_index", [0, 1, 2])
    @pytest.mark.parametrize("speed_mps", [2.0, 8.0, 16.0])
    def test_shifted_label_equals_the_unshifted_one(
        self,
        shift_index: int,
        speed_mps: float,
    ) -> None:
        waypoints, yaws = _straight_line(speed_mps, _HORIZON + 2)
        shifted, shifted_yaws = shifted_planning_label(
            waypoints, yaws, shift_index, _HORIZON,
        )
        expected, expected_yaws = shifted_planning_label(
            waypoints, yaws, 0, _HORIZON,
        )
        assert np.allclose(shifted, expected)
        assert np.allclose(shifted_yaws, expected_yaws)

    def test_without_re_anchoring_the_lag_would_be_visible(self) -> None:
        """Guards the reason re-anchoring exists rather than just slicing.

        A plain slice leaves the plan expressed around a pose the ego has
        already left, which is exactly the constant offset the transform is
        there to remove. If this ever stops being true the test above has
        stopped testing anything.
        """
        waypoints, yaws = _straight_line(8.0, _HORIZON + 2)
        naive_slice = waypoints[2 : 2 + _HORIZON]
        re_anchored, _ = shifted_planning_label(waypoints, yaws, 2, _HORIZON)
        offset = np.abs(naive_slice - re_anchored).max()
        assert offset > 1.0


class TestTurning:
    """On a curve the shift is a real rotation, not just a translation."""

    def test_re_anchored_plan_starts_ahead_and_stays_finite(self) -> None:
        waypoints, yaws = _arc(radius_m=20.0, speed_mps=8.0, num_ticks=_HORIZON + 2)
        shifted, shifted_yaws = shifted_planning_label(waypoints, yaws, 2, _HORIZON)
        assert np.isfinite(shifted).all()
        assert np.isfinite(shifted_yaws).all()
        # The plan is expressed from the execution pose, so its first waypoint
        # is one stride ahead of that pose rather than one stride ahead of the
        # anchor the ego has already left.
        first_step = np.linalg.norm(shifted[0])
        assert first_step == pytest.approx(8.0 * _STRIDE_S, rel=0.02)

    def test_yaws_are_relative_to_the_execution_pose(self) -> None:
        waypoints, yaws = _arc(radius_m=20.0, speed_mps=8.0, num_ticks=_HORIZON + 2)
        _, shifted_yaws = shifted_planning_label(waypoints, yaws, 2, _HORIZON)
        # Turning consistently one way, every remaining yaw is on that side of
        # the execution heading and none has wrapped.
        assert bool((shifted_yaws > 0.0).all())
        assert bool((np.abs(shifted_yaws) <= np.pi).all())


class TestContract:
    """What the function refuses, so a misconfiguration fails at the sample."""

    def test_zero_shift_is_the_leading_window(self) -> None:
        waypoints, yaws = _straight_line(8.0, _HORIZON + 2)
        shifted, shifted_yaws = shifted_planning_label(waypoints, yaws, 0, _HORIZON)
        assert np.array_equal(shifted, waypoints[:_HORIZON])
        assert np.array_equal(shifted_yaws, yaws[:_HORIZON])

    def test_output_length_is_always_the_horizon(self) -> None:
        waypoints, yaws = _straight_line(8.0, _HORIZON + 2)
        for shift_index in (0, 1, 2):
            shifted, shifted_yaws = shifted_planning_label(
                waypoints, yaws, shift_index, _HORIZON,
            )
            assert len(shifted) == _HORIZON
            assert len(shifted_yaws) == _HORIZON

    def test_a_shift_the_window_cannot_reach_is_refused(self) -> None:
        waypoints, yaws = _straight_line(8.0, _HORIZON)
        with pytest.raises(ValueError, match="future_ego_pose_extra_ticks"):
            shifted_planning_label(waypoints, yaws, 1, _HORIZON)

    def test_a_negative_shift_is_refused(self) -> None:
        waypoints, yaws = _straight_line(8.0, _HORIZON + 2)
        with pytest.raises(ValueError, match="must not be negative"):
            shifted_planning_label(waypoints, yaws, -1, _HORIZON)


class TestConfigInvariants:
    """Reading further ahead must not change what the model predicts."""

    def test_extra_ticks_widen_the_read_but_not_the_head(self) -> None:
        from lead.config import LeadConfig

        plain = LeadConfig()
        widened = LeadConfig()
        stride = plain.policy.transfuser.future_ego_pose_iterations[0]
        widened.policy.transfuser.future_ego_pose_extra_ticks = 2 * stride

        assert len(widened.policy.transfuser.future_ego_pose_iterations) == len(
            plain.policy.transfuser.future_ego_pose_iterations,
        ) + 2
        assert (
            widened.policy.transfuser.num_ego_pose_prediction
            == plain.policy.transfuser.num_ego_pose_prediction
        )

    def test_a_shift_between_label_ticks_is_refused(self) -> None:
        from lead.config import LeadConfig

        config = LeadConfig()
        config.policy.transfuser.future_ego_pose_extra_ticks = 1
        with pytest.raises(ValueError, match="not a multiple"):
            _ = config.policy.transfuser.future_ego_pose_iterations

    def test_the_default_config_reads_exactly_what_it_predicts(self) -> None:
        from lead.config import LeadConfig

        transfuser = LeadConfig().policy.transfuser
        assert transfuser.future_ego_pose_extra_ticks == 0
        assert len(transfuser.future_ego_pose_iterations) == (
            transfuser.num_ego_pose_prediction
        )


class TestShiftDraw:
    """The severity sets the ceiling; the draw spreads under it."""

    def test_zero_severity_never_shifts(self) -> None:
        generator = np.random.default_rng(0)
        drawn = {sample_shift_index(0.0, 2, generator) for _ in range(50)}
        assert drawn == {0}

    def test_full_severity_reaches_the_ceiling_and_zero(self) -> None:
        generator = np.random.default_rng(0)
        drawn = {sample_shift_index(1.0, 2, generator) for _ in range(200)}
        assert drawn == {0, 1, 2}

    def test_draw_never_exceeds_the_ceiling(self) -> None:
        generator = np.random.default_rng(0)
        for _ in range(200):
            severity = float(generator.random())
            assert 0 <= sample_shift_index(severity, 2, generator) <= 2
