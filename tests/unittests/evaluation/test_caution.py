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
    corridor_mask,
    observability_caution,
    surrogate_risk_event,
    target_speed_multiplier,
)

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
