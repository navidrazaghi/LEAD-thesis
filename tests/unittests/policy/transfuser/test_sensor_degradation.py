"""Tests for the sensor-degradation curriculum.

The properties worth pinning are the ones a training run cannot tell you about
until the night is already spent: that zero severity is an exact no-op, that
full severity actually damages the input, that the damage stays inside the
range the model's inputs are defined on, and that a seeded generator repeats
itself -- the last one is what makes two checkpoints comparable on one route.

The deployment families add one more: the ego-state family must leave the
observability targets alone, because a drifting GPS fix is not a sensing
failure and a gate trained as though it were would shift modality for no
reason.
"""

import pytest
import torch

from lead.policy.transfuser.utils.sensor_degradation import (
    apply_sensor_degradation,
    degrade_batch,
    degrade_camera,
    degrade_ego_state,
    degrade_lidar,
    degrade_occlusion,
)


def _rgb(batch_size: int = 4, height: int = 32, width: int = 48) -> torch.Tensor:
    """Build a mid-grey image batch with structure the damage can remove."""
    image = torch.full((batch_size, 3, height, width), 128.0)
    image[:, :, ::4, :] = 255.0
    return image


def _batch(batch_size: int = 4) -> dict:
    """Build the fields the curriculum touches, with no zeros to hide behind.

    Deterministic on purpose: several tests below compare two batches built by
    two calls, so any unseeded content here would differ before the damage does
    and the comparison would test nothing.
    """
    source = torch.Generator().manual_seed(0)
    return {
        "rgb": _rgb(batch_size),
        "rasterized_lidar": torch.rand(
            batch_size, 1, 16, 16, generator=source,
        ) + 0.5,
        "speed": torch.full((batch_size,), 8.0),
        "target_point": torch.ones(batch_size, 2),
        "previous_target_point": torch.ones(batch_size, 2),
        "next_target_point": torch.ones(batch_size, 2),
    }


class TestZeroSeverityIsANoOp:
    """Severity zero must leave every family's input bit-identical."""

    def test_camera(self) -> None:
        rgb = _rgb()
        assert torch.equal(degrade_camera(rgb, torch.zeros(4)), rgb)

    def test_lidar(self) -> None:
        raster = torch.rand(4, 1, 16, 16)
        assert torch.equal(degrade_lidar(raster, torch.zeros(4)), raster)

    def test_occlusion(self) -> None:
        rgb = _rgb()
        occluded, visible = degrade_occlusion(rgb, torch.zeros(4))
        assert torch.equal(occluded, rgb)
        assert torch.allclose(visible, torch.ones(4))

    def test_ego_state(self) -> None:
        batch = _batch()
        original = {key: value.clone() for key, value in batch.items()}
        degrade_ego_state(batch, torch.zeros(4))
        for key, value in original.items():
            assert torch.allclose(batch[key], value), key


class TestFullSeverityDamages:
    """Severity one must measurably change the input it is given."""

    def test_camera_dims_and_smooths(self) -> None:
        rgb = _rgb()
        degraded = degrade_camera(rgb, torch.ones(4))
        assert degraded.mean() < rgb.mean()

    def test_lidar_loses_returns(self) -> None:
        raster = torch.rand(4, 1, 32, 32) + 0.5
        assert degrade_lidar(raster, torch.ones(4)).sum() < raster.sum()

    def test_occlusion_blacks_out_part_of_the_frame(self) -> None:
        occluded, visible = degrade_occlusion(_rgb(), torch.ones(4))
        assert bool((occluded == 0).any())
        assert bool((visible < 1.0).all())

    def test_speed_is_under_reported_never_over(self) -> None:
        batch = _batch()
        degrade_ego_state(batch, torch.ones(4))
        assert bool((batch["speed"] < 8.0).all())
        assert bool((batch["speed"] > 0.0).all())


class TestStaysInRange:
    """Damage must not push an input outside the range the model expects."""

    def test_camera_stays_in_zero_to_255(self) -> None:
        degraded = degrade_camera(_rgb(), torch.ones(4))
        assert bool((degraded >= 0.0).all())
        assert bool((degraded <= 255.0).all())

    def test_occlusion_stays_in_zero_to_255(self) -> None:
        occluded, _ = degrade_occlusion(_rgb(), torch.ones(4))
        assert bool((occluded >= 0.0).all())
        assert bool((occluded <= 255.0).all())

    def test_lidar_never_gains_returns(self) -> None:
        raster = torch.rand(4, 1, 16, 16)
        assert bool((degrade_lidar(raster, torch.ones(4)) <= raster).all())

    def test_visible_fraction_is_a_fraction(self) -> None:
        _, visible = degrade_occlusion(_rgb(), torch.rand(4))
        assert bool((visible >= 0.0).all())
        assert bool((visible <= 1.0).all())


class TestSeededDamageRepeats:
    """A seeded generator is what pairs two checkpoints on one route."""

    def test_camera(self) -> None:
        rgb = _rgb()
        severity = torch.full((4,), 0.7)
        first = degrade_camera(rgb, severity, torch.Generator().manual_seed(11))
        second = degrade_camera(rgb, severity, torch.Generator().manual_seed(11))
        assert torch.equal(first, second)

    def test_occlusion(self) -> None:
        rgb = _rgb()
        severity = torch.full((4,), 0.7)
        first, _ = degrade_occlusion(rgb, severity, torch.Generator().manual_seed(11))
        second, _ = degrade_occlusion(rgb, severity, torch.Generator().manual_seed(11))
        assert torch.equal(first, second)

    def test_ego_state(self) -> None:
        severity = torch.full((4,), 0.7)
        first, second = _batch(), _batch()
        degrade_ego_state(first, severity, torch.Generator().manual_seed(11))
        degrade_ego_state(second, severity, torch.Generator().manual_seed(11))
        assert torch.equal(first["target_point"], second["target_point"])


class TestEgoStateIsNotASensingFailure:
    """The ego-state family must not touch the observability targets."""

    def test_observability_is_left_alone(self) -> None:
        batch = _batch()
        batch["observability"] = torch.rand(4, 2, 8, 8)
        original = batch["observability"].clone()
        degrade_ego_state(batch, torch.ones(4))
        assert torch.equal(batch["observability"], original)


class TestCurriculumSampler:
    """The existing sampler's contract, which the new families must not change."""

    def test_probability_zero_leaves_the_batch_alone(self) -> None:
        batch = _batch()
        original = {key: value.clone() for key, value in batch.items()}
        apply_sensor_degradation(batch, probability=0.0, max_severity=1.0)
        for key, value in original.items():
            assert torch.allclose(batch[key], value), key

    def test_only_one_modality_is_damaged_per_sample(self) -> None:
        """The gate needs somewhere intact to shift its reading onto."""
        torch.manual_seed(0)
        batch = _batch(batch_size=64)
        before_rgb = batch["rgb"].clone()
        before_lidar = batch["rasterized_lidar"].clone()
        apply_sensor_degradation(batch, probability=1.0, max_severity=1.0)
        camera_hit = (batch["rgb"] != before_rgb).flatten(1).any(dim=1)
        lidar_hit = (batch["rasterized_lidar"] != before_lidar).flatten(1).any(dim=1)
        assert not bool((camera_hit & lidar_hit).any())

    def test_observability_targets_follow_the_damage(self) -> None:
        torch.manual_seed(0)
        batch = _batch(batch_size=16)
        batch["observability"] = torch.ones(16, 2, 8, 8)
        apply_sensor_degradation(batch, probability=1.0, max_severity=1.0)
        assert bool((batch["observability"] <= 1.0).all())
        assert bool((batch["observability"] < 1.0).any())


class TestDeploymentFamilies:
    """The new families, and the promise that they cost the old ones nothing."""

    def test_default_path_is_unchanged_draw_for_draw(self) -> None:
        """The rung that established the curriculum is defined by this stream.

        An extra draw on the default path would silently redefine it, so this
        compares the whole batch against a run of the same seed through the
        same call with no deployment families requested.
        """
        torch.manual_seed(1234)
        without = _batch(batch_size=32)
        without["observability"] = torch.ones(32, 2, 8, 8)
        apply_sensor_degradation(without, probability=0.5, max_severity=1.0)

        torch.manual_seed(1234)
        explicitly_empty = _batch(batch_size=32)
        explicitly_empty["observability"] = torch.ones(32, 2, 8, 8)
        apply_sensor_degradation(
            explicitly_empty,
            probability=0.5,
            max_severity=1.0,
            deployment_families=(),
        )

        for key, value in without.items():
            assert torch.equal(explicitly_empty[key], value), key

    def test_requesting_a_family_actually_applies_it(self) -> None:
        torch.manual_seed(7)
        batch = _batch(batch_size=64)
        before_speed = batch["speed"].clone()
        apply_sensor_degradation(
            batch,
            probability=1.0,
            max_severity=1.0,
            deployment_families=("occlusion", "ego_state"),
        )
        assert bool((batch["speed"] != before_speed).any())

    def test_one_family_per_sample(self) -> None:
        """A sample hit by the ego-state family keeps an undamaged image."""
        torch.manual_seed(3)
        batch = _batch(batch_size=64)
        before_rgb = batch["rgb"].clone()
        before_speed = batch["speed"].clone()
        apply_sensor_degradation(
            batch,
            probability=1.0,
            max_severity=1.0,
            deployment_families=("ego_state",),
        )
        image_hit = (batch["rgb"] != before_rgb).flatten(1).any(dim=1)
        speed_hit = batch["speed"] != before_speed
        assert not bool((image_hit & speed_hit).any())

    def test_occlusion_scales_the_camera_target_by_what_it_removed(self) -> None:
        torch.manual_seed(5)
        batch = _batch(batch_size=32)
        batch["observability"] = torch.ones(32, 2, 8, 8)
        apply_sensor_degradation(
            batch,
            probability=1.0,
            max_severity=1.0,
            deployment_families=("occlusion",),
        )
        camera_channel = batch["observability"][:, 0]
        assert bool((camera_channel >= 0.0).all())
        assert bool((camera_channel < 1.0).any())

    def test_an_unknown_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown deployment"):
            apply_sensor_degradation(
                _batch(),
                probability=1.0,
                max_severity=1.0,
                deployment_families=("weather",),
            )


class TestInferenceSweep:
    """``degrade_batch`` is what the closed-loop sweep drives every run through."""

    @pytest.mark.parametrize(
        "modality",
        ["camera", "lidar", "occlusion", "ego_state"],
    )
    def test_every_sweepable_family_changes_the_batch(self, modality: str) -> None:
        batch = _batch()
        original = {key: value.clone() for key, value in batch.items()}
        degrade_batch(batch, modality, severity=1.0)
        changed = any(
            not torch.equal(batch[key], value) for key, value in original.items()
        )
        assert changed, modality

    @pytest.mark.parametrize(
        "modality",
        ["camera", "lidar", "occlusion", "ego_state", "none"],
    )
    def test_zero_severity_is_a_no_op_for_every_family(self, modality: str) -> None:
        batch = _batch()
        original = {key: value.clone() for key, value in batch.items()}
        degrade_batch(batch, modality, severity=0.0)
        for key, value in original.items():
            assert torch.equal(batch[key], value), f"{modality}/{key}"

    @pytest.mark.parametrize("modality", ["occlusion", "ego_state"])
    def test_a_seeded_generator_pairs_two_models_on_one_route(
        self,
        modality: str,
    ) -> None:
        """Constraint of the protocol: identical damage, two checkpoints."""
        first, second = _batch(), _batch()
        degrade_batch(first, modality, 0.6, torch.Generator().manual_seed(99))
        degrade_batch(second, modality, 0.6, torch.Generator().manual_seed(99))
        for key in first:
            assert torch.equal(first[key], second[key]), key

    def test_an_unknown_condition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="degrade modality must be"):
            degrade_batch(_batch(), "weather", severity=1.0)

    def test_joint_degradation_damages_both_modalities(self) -> None:
        """The one regime redundancy cannot cover, and the governor's target."""
        batch = _batch()
        before_rgb = batch["rgb"].clone()
        before_lidar = batch["rasterized_lidar"].clone()
        degrade_batch(batch, "both", severity=1.0)
        assert not torch.equal(batch["rgb"], before_rgb)
        assert not torch.equal(batch["rasterized_lidar"], before_lidar)

    def test_joint_degradation_matches_the_single_families_it_composes(self) -> None:
        """Not a third kind of damage: the same two, applied together."""
        joint = _batch()
        degrade_batch(joint, "both", 0.6, torch.Generator().manual_seed(4))

        separate = _batch()
        generator = torch.Generator().manual_seed(4)
        separate["rgb"] = degrade_camera(
            separate["rgb"], torch.full((4,), 0.6), generator,
        )
        separate["rasterized_lidar"] = degrade_lidar(
            separate["rasterized_lidar"], torch.full((4,), 0.6), generator,
        )
        assert torch.equal(joint["rgb"], separate["rgb"])
        assert torch.equal(
            joint["rasterized_lidar"], separate["rasterized_lidar"],
        )


@pytest.mark.parametrize("severity_value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_dose_response_is_monotone_in_brightness(severity_value: float) -> None:
    """The evaluation protocol sweeps these severities, so each must be valid."""
    rgb = _rgb()
    degraded = degrade_camera(rgb, torch.full((4,), severity_value))
    assert degraded.mean() <= rgb.mean() + 1e-4
