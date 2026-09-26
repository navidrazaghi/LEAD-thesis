"""Tests for the caution signal and the speed it buys.

Three properties carry the design. The signal must fall to zero when the model
says it resolves the road and rise to one when it says it does not. It must read
redundancy the way two sensors are meant to be read, so one working modality is
enough. And the mapping must leave the frozen policy exactly as it was whenever
the calibrator has not asked for anything, because the ablation this feeds
compares a mechanism against its own absence -- if the un-governed arm is not
reproduced bit for bit, it compares two tunings instead.
"""

import pytest
import torch

from lead.config import LeadConfig
from lead.evaluation.inference.caution import (
    cell_camera_geometry,
    corridor_mask,
    cross_modal_caution,
    observability_caution,
    surrogate_risk_event,
    target_speed_multiplier,
)
from lead.policy.transfuser.encoder.fusion_geometry import bev_cell_centres

_CELL_HEIGHT = 16
_CELL_WIDTH = 16


@pytest.fixture(name="config")
def _config() -> LeadConfig:
    return LeadConfig()


def _logits(value: float, shape: tuple[int, int, int, int]) -> torch.Tensor:
    """A constant observability field, given as the logit of a probability."""
    probability = torch.full(shape, value)
    return torch.log(probability / (1.0 - probability))


class TestSignal:
    """What the caution number means."""

    def test_a_fully_resolved_corridor_gives_no_caution(self, config: LeadConfig) -> None:
        logits = _logits(0.999, (1, 2, _CELL_HEIGHT, _CELL_WIDTH))
        assert observability_caution(logits, config) == pytest.approx(0.0, abs=1e-2)

    def test_an_unresolved_corridor_gives_full_caution(self, config: LeadConfig) -> None:
        logits = _logits(0.001, (1, 2, _CELL_HEIGHT, _CELL_WIDTH))
        assert observability_caution(logits, config) == pytest.approx(1.0, abs=1e-2)

    def test_it_is_monotone_in_how_well_the_road_is_resolved(
        self,
        config: LeadConfig,
    ) -> None:
        previous = 1.1
        for resolved in (0.1, 0.3, 0.5, 0.7, 0.9):
            caution = observability_caution(
                _logits(resolved, (1, 2, _CELL_HEIGHT, _CELL_WIDTH)),
                config,
            )
            assert caution < previous
            previous = caution

    def test_one_working_modality_is_enough(self, config: LeadConfig) -> None:
        """Redundancy is why two sensors are carried; averaging would deny it."""
        logits = torch.empty(1, 2, _CELL_HEIGHT, _CELL_WIDTH)
        logits[:, 0] = _logits(0.99, (1, _CELL_HEIGHT, _CELL_WIDTH))
        logits[:, 1] = _logits(0.01, (1, _CELL_HEIGHT, _CELL_WIDTH))
        caution = observability_caution(logits, config)
        assert caution == pytest.approx(0.0, abs=1e-2)

    def test_the_signal_is_a_fraction(self, config: LeadConfig) -> None:
        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            logits = torch.randn(
                1, 2, _CELL_HEIGHT, _CELL_WIDTH, generator=generator,
            ) * 5.0
            caution = observability_caution(logits, config)
            assert 0.0 <= caution <= 1.0

    def test_damage_confined_behind_the_ego_does_not_raise_caution(
        self,
        config: LeadConfig,
    ) -> None:
        """A mean over the whole grid would report caution here; this must not."""
        mask = corridor_mask(config, _CELL_HEIGHT, _CELL_WIDTH, torch.device("cpu"))
        logits = _logits(0.99, (1, 2, _CELL_HEIGHT, _CELL_WIDTH))
        logits[..., ~mask] = _logits(0.01, (1,)).item()
        assert observability_caution(logits, config) == pytest.approx(0.0, abs=1e-2)


class TestModalityRule:
    """Which combination rule is used, and what each one costs.

    The default reports almost nothing under single-modality damage, and that
    is the intended behaviour rather than a defect: the trained rungs drive no
    worse under full camera destruction than intact, so one working sensor is
    enough and slowing there would spend route completion on nothing. These pin
    the trade-off down so a change to the default has to be deliberate.
    """

    @staticmethod
    def _one_modality_destroyed() -> torch.Tensor:
        logits = torch.empty(1, 2, _CELL_HEIGHT, _CELL_WIDTH)
        logits[:, 0] = _logits(0.105, (1, _CELL_HEIGHT, _CELL_WIDTH))
        logits[:, 1] = _logits(0.948, (1, _CELL_HEIGHT, _CELL_WIDTH))
        return logits

    def test_the_default_rule_stays_quiet_when_one_sensor_still_sees(
        self,
        config: LeadConfig,
    ) -> None:
        assert config.evaluation.inference.caution_modality_rule == "best"
        caution = observability_caution(self._one_modality_destroyed(), config)
        assert caution < 0.1

    @pytest.mark.parametrize(
        ("rule", "lower", "upper"),
        [("mean", 0.4, 0.6), ("worst", 0.8, 1.0)],
    )
    def test_the_other_rules_react_to_a_single_failed_sensor(
        self,
        config: LeadConfig,
        rule: str,
        lower: float,
        upper: float,
    ) -> None:
        config.evaluation.inference.caution_modality_rule = rule
        caution = observability_caution(self._one_modality_destroyed(), config)
        assert lower <= caution <= upper

    def test_every_rule_reacts_when_both_sensors_fail(
        self,
        config: LeadConfig,
    ) -> None:
        """Joint degradation is the regime the default governor is built for."""
        logits = _logits(0.05, (1, 2, _CELL_HEIGHT, _CELL_WIDTH))
        for rule in ("best", "mean", "worst"):
            config.evaluation.inference.caution_modality_rule = rule
            assert observability_caution(logits, config) > 0.9

    def test_an_unknown_rule_is_refused(self, config: LeadConfig) -> None:
        config.evaluation.inference.caution_modality_rule = "median"
        with pytest.raises(ValueError, match="caution_modality_rule"):
            observability_caution(
                _logits(0.5, (1, 2, _CELL_HEIGHT, _CELL_WIDTH)),
                config,
            )


class TestCorridor:
    """The region the signal is measured over."""

    def test_it_selects_some_cells_but_not_all(self, config: LeadConfig) -> None:
        mask = corridor_mask(config, _CELL_HEIGHT, _CELL_WIDTH, torch.device("cpu"))
        assert bool(mask.any())
        assert not bool(mask.all())

    def test_the_selected_cells_are_the_ones_actually_ahead(
        self,
        config: LeadConfig,
    ) -> None:
        """Checked against independently computed cell centres, not the mask.

        Columns run along x and rows along y. A mask built on the transposed
        convention stays non-empty and passes every symmetric check while
        pointing across the road instead of along it, so the only test that
        catches it is one that asks where the selected cells actually are.
        """
        transfuser = config.policy.transfuser
        rows = transfuser.lidar_bev_grid_rows
        cols = transfuser.lidar_bev_grid_cols
        centres = torch.as_tensor(
            bev_cell_centres(config), dtype=torch.float32,
        ).reshape(rows, cols, 2)

        mask = corridor_mask(config, rows, cols, torch.device("cpu"))
        assert bool(mask.any()), "the corridor selected nothing"

        governor = config.evaluation.inference
        selected = centres[mask]
        assert bool((selected[:, 0] >= 0.0).all()), "a selected cell is behind the ego"
        assert bool(
            (selected[:, 0] <= governor.caution_corridor_length_meter).all(),
        ), "a selected cell is beyond the corridor's length"
        assert bool(
            (selected[:, 1].abs() <= governor.caution_corridor_half_width_meter).all(),
        ), "a selected cell is outside the corridor's width"

    def test_it_selects_every_cell_that_qualifies(self, config: LeadConfig) -> None:
        """The complement of the test above: nothing in the corridor is missed."""
        transfuser = config.policy.transfuser
        rows = transfuser.lidar_bev_grid_rows
        cols = transfuser.lidar_bev_grid_cols
        centres = torch.as_tensor(
            bev_cell_centres(config), dtype=torch.float32,
        ).reshape(rows, cols, 2)
        governor = config.evaluation.inference

        expected = (
            (centres[..., 0] >= 0.0)
            & (centres[..., 0] <= governor.caution_corridor_length_meter)
            & (centres[..., 1].abs() <= governor.caution_corridor_half_width_meter)
        )
        mask = corridor_mask(config, rows, cols, torch.device("cpu"))
        assert torch.equal(mask, expected)

    def test_a_corridor_off_the_grid_reports_maximum_caution(
        self,
        config: LeadConfig,
    ) -> None:
        """Dividing by an empty corridor would report perfect confidence."""
        config.evaluation.inference.caution_corridor_half_width_meter = 0.0
        config.evaluation.inference.caution_corridor_length_meter = -1.0
        logits = _logits(0.99, (1, 2, _CELL_HEIGHT, _CELL_WIDTH))
        assert observability_caution(logits, config) == pytest.approx(1.0)


class TestMapping:
    """Caution and the calibrated scalar into a speed factor."""

    def test_a_zero_scalar_leaves_the_policy_exactly_as_it_was(
        self,
        config: LeadConfig,
    ) -> None:
        for caution in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert target_speed_multiplier(caution, 0.0, config) == 1.0

    def test_no_caution_leaves_the_policy_exactly_as_it_was(
        self,
        config: LeadConfig,
    ) -> None:
        for scalar in (0.0, 0.5, 1.0):
            assert target_speed_multiplier(0.0, scalar, config) == 1.0

    def test_the_floor_is_never_breached(self, config: LeadConfig) -> None:
        floor = config.evaluation.inference.caution_speed_floor
        assert target_speed_multiplier(1.0, 1.0, config) == pytest.approx(floor)
        assert target_speed_multiplier(5.0, 5.0, config) == pytest.approx(floor)

    def test_more_caution_never_means_more_speed(self, config: LeadConfig) -> None:
        previous = 1.1
        for caution in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            multiplier = target_speed_multiplier(caution, 0.8, config)
            assert multiplier <= previous
            previous = multiplier


class TestCrossModalGeometry:
    """The projection table the label-free signal is built on."""

    def test_it_covers_every_bev_token(self, config: LeadConfig) -> None:
        transfuser = config.policy.transfuser
        count = transfuser.lidar_bev_grid_rows * transfuser.lidar_bev_grid_cols
        geometry = cell_camera_geometry(config)
        assert geometry.image_xy.shape == (count, 2)
        assert geometry.axial_depth.shape == (count,)
        assert geometry.visible.shape == (count,)

    def test_some_cells_are_seen_and_some_are_not(self, config: LeadConfig) -> None:
        """Cells behind the rig have no projection; that must be recorded."""
        geometry = cell_camera_geometry(config)
        assert bool(geometry.visible.any())
        assert not bool(geometry.visible.all())

    def test_visible_cells_land_inside_the_stitched_image(
        self,
        config: LeadConfig,
    ) -> None:
        geometry = cell_camera_geometry(config)
        seen = geometry.image_xy[geometry.visible]
        assert bool((seen >= 0.0).all())
        assert bool((seen <= 1.0).all())

    def test_depth_grows_with_distance_ahead(self, config: LeadConfig) -> None:
        """The axial depth has to track the cell's actual range, or the
        comparison against the depth head is comparing nothing."""
        transfuser = config.policy.transfuser
        rows = transfuser.lidar_bev_grid_rows
        cols = transfuser.lidar_bev_grid_cols
        centres = torch.as_tensor(
            bev_cell_centres(config), dtype=torch.float32,
        )
        geometry = cell_camera_geometry(config)
        corridor = corridor_mask(config, rows, cols, torch.device("cpu")).reshape(-1)
        chosen = corridor & geometry.visible
        forward = centres[chosen, 0]
        depth = geometry.axial_depth[chosen]
        assert len(forward) >= 2
        # Not equality: the cameras sit off the ego origin and off the axis, so
        # the two differ by the mounting offset rather than being the same
        # number.
        order_by_range = torch.argsort(forward)
        assert bool((torch.diff(depth[order_by_range]) >= -1.0).all())


class TestCrossModalCaution:
    """What the label-free signal reports, and when."""

    @staticmethod
    def _inputs(config: LeadConfig, depth_value: float, lidar_value: float):
        transfuser = config.policy.transfuser
        depth = torch.full(
            (1, transfuser.final_image_height // 8, transfuser.final_image_width // 8),
            depth_value,
        )
        lidar = torch.full((1, 1, 64, 64), lidar_value)
        return depth, lidar

    def test_agreement_reports_no_caution(self, config: LeadConfig) -> None:
        """Empty road: LiDAR returns nothing and the camera sees far.

        With no returns and the camera predicting depth well beyond every cell
        in the corridor, neither contradiction fires.
        """
        depth, lidar = self._inputs(config, depth_value=45.0, lidar_value=0.0)
        assert cross_modal_caution(depth, lidar, config) == pytest.approx(0.0)

    def test_lidar_returns_the_camera_sees_through_are_caught(
        self,
        config: LeadConfig,
    ) -> None:
        """What a dimmed or blurred camera looks like against an intact sweep."""
        depth, lidar = self._inputs(config, depth_value=45.0, lidar_value=1.0)
        assert cross_modal_caution(depth, lidar, config) > 0.9

    def test_a_surface_the_lidar_lost_is_caught(self, config: LeadConfig) -> None:
        """What dropout looks like against an intact camera.

        The camera puts a surface at a corridor cell's own range while the
        sweep returns nothing there.
        """
        transfuser = config.policy.transfuser
        rows = transfuser.lidar_bev_grid_rows
        cols = transfuser.lidar_bev_grid_cols
        geometry = cell_camera_geometry(config)
        corridor = corridor_mask(config, rows, cols, torch.device("cpu")).reshape(-1)
        chosen = corridor & geometry.visible
        typical = float(geometry.axial_depth[chosen].median())

        depth, lidar = self._inputs(config, depth_value=typical, lidar_value=0.0)
        assert cross_modal_caution(depth, lidar, config) > 0.0

    def test_the_signal_is_a_fraction(self, config: LeadConfig) -> None:
        for depth_value in (5.0, 15.0, 30.0, 45.0):
            for lidar_value in (0.0, 1.0):
                depth, lidar = self._inputs(config, depth_value, lidar_value)
                caution = cross_modal_caution(depth, lidar, config)
                assert 0.0 <= caution <= 1.0

    def test_it_reacts_to_single_modality_damage(self, config: LeadConfig) -> None:
        """The whole reason this signal exists alongside the trained head.

        The observability signal reports per modality and combines by the
        better of the two, so one destroyed sensor leaves it silent. This one
        is a comparison between the two, so a single failure is exactly what
        it sees.
        """
        agreeing, lidar = self._inputs(config, depth_value=45.0, lidar_value=0.0)
        quiet = cross_modal_caution(agreeing, lidar, config)

        broken, intact_lidar = self._inputs(config, depth_value=45.0, lidar_value=1.0)
        loud = cross_modal_caution(broken, intact_lidar, config)
        assert loud - quiet > 0.5


class TestSurrogateRisk:
    """What the calibrator counts against."""

    def test_speed_into_an_unresolved_corridor_counts(self, config: LeadConfig) -> None:
        assert surrogate_risk_event(0.9, 10.0, config)

    def test_a_resolved_corridor_does_not_count_at_any_speed(
        self,
        config: LeadConfig,
    ) -> None:
        assert not surrogate_risk_event(0.0, 30.0, config)

    def test_standing_still_does_not_count_however_blind(
        self,
        config: LeadConfig,
    ) -> None:
        """A stopped car in an unresolved scene is the governor working."""
        assert not surrogate_risk_event(1.0, 0.0, config)
