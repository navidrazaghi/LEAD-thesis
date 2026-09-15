"""Tests for the per-modality observability targets and head."""

import numpy as np
import pytest
import torch

from lead.config import LeadConfig, load_lead_config
from lead.policy.transfuser.dataloader.observability import (
    NUM_OBSERVABILITY_CHANNELS,
    ObservabilityChannel,
    build_observability_targets,
)
from lead.policy.transfuser.decoder.observability_decoder import ObservabilityDecoder


@pytest.fixture
def lead_config() -> LeadConfig:
    """Fixture providing the root config tree."""
    return load_lead_config()


def make_box(
    position: tuple[float, float],
    *,
    box_class: str = "car",
    visible_pixels: int = 100,
    num_points: int = 100,
    extent: tuple[float, float] = (2.2, 0.9),
    yaw: float = 0.0,
) -> dict:
    """Build a view-frame box dict of the shape the label builder reads."""
    return {
        "class": box_class,
        "position": [position[0], position[1], 0.0],
        "extent": [extent[0], extent[1], 0.75],
        "yaw": yaw,
        "visible_pixels": visible_pixels,
        "num_points": num_points,
    }


def cell_of(lead_config: LeadConfig, x: float, y: float) -> tuple[int, int]:
    """The ``(row, col)`` observability cell an ego-frame point falls in."""
    config = lead_config.policy.transfuser
    rows = config.lidar_height_pixel // config.bev_downsample_factor
    cols = config.lidar_width_pixel // config.bev_downsample_factor
    col = int(
        (x - config.bev_min_x_meter)
        / ((config.bev_max_x_meter - config.bev_min_x_meter) / cols),
    )
    row = int(
        (y - config.bev_min_y_meter)
        / ((config.bev_max_y_meter - config.bev_min_y_meter) / rows),
    )
    return row, col


class TestTargetShape:
    """Tests for the grid the targets live on."""

    def test_matches_the_center_net_cell_grid(self, lead_config: LeadConfig) -> None:
        config = lead_config.policy.transfuser
        built = build_observability_targets([], lead_config)

        expected = (
            NUM_OBSERVABILITY_CHANNELS,
            config.lidar_height_pixel // config.bev_downsample_factor,
            config.lidar_width_pixel // config.bev_downsample_factor,
        )
        assert built["observability"].shape == expected
        assert built["observability_mask"].shape == expected
        assert built["observability"].dtype == np.float32
        assert built["observability_mask"].dtype == np.float32

    def test_no_boxes_supervises_nothing(self, lead_config: LeadConfig) -> None:
        built = build_observability_targets([], lead_config)
        assert built["observability_mask"].sum() == 0.0
        assert built["observability"].sum() == 0.0


class TestRasterization:
    """Tests for where a box's measurement lands on the grid."""

    def test_a_box_supervises_the_cell_it_occupies(
        self,
        lead_config: LeadConfig,
    ) -> None:
        built = build_observability_targets(
            [make_box((20.0, 4.0))],
            lead_config,
        )
        row, col = cell_of(lead_config, 20.0, 4.0)

        assert built["observability_mask"][:, row, col].all()
        np.testing.assert_allclose(built["observability"][:, row, col], 1.0)

    def test_cells_the_box_misses_stay_unsupervised(
        self,
        lead_config: LeadConfig,
    ) -> None:
        built = build_observability_targets(
            [make_box((20.0, 4.0))],
            lead_config,
        )
        far_row, far_col = cell_of(lead_config, -20.0, -30.0)
        assert built["observability_mask"][:, far_row, far_col].sum() == 0.0

    def test_a_rotated_box_covers_its_axis_aligned_extent(
        self,
        lead_config: LeadConfig,
    ) -> None:
        # Turned broadside, a 4.4 m long box reaches further across the rows
        # than the same box pointing along x.
        along = build_observability_targets(
            [make_box((20.0, 0.0), yaw=0.0)],
            lead_config,
        )
        across = build_observability_targets(
            [make_box((20.0, 0.0), yaw=np.pi / 2)],
            lead_config,
        )
        rows_along = along["observability_mask"][ObservabilityChannel.LIDAR].any(axis=1)
        rows_across = across["observability_mask"][ObservabilityChannel.LIDAR].any(
            axis=1,
        )
        assert rows_across.sum() > rows_along.sum()

    def test_boxes_outside_the_crop_are_dropped(
        self,
        lead_config: LeadConfig,
    ) -> None:
        config = lead_config.policy.transfuser
        built = build_observability_targets(
            [make_box((config.bev_max_x_meter + 50.0, 0.0))],
            lead_config,
        )
        assert built["observability_mask"].sum() == 0.0


class TestMeasurementSemantics:
    """Tests for how the expert's counts become a target."""

    def test_saturates_at_the_experts_own_threshold(
        self,
        lead_config: LeadConfig,
    ) -> None:
        occlusion = lead_config.expert.occlusion
        row, col = cell_of(lead_config, 20.0, 0.0)
        built = build_observability_targets(
            [
                make_box(
                    (20.0, 0.0),
                    visible_pixels=occlusion.vehicle_min_num_visible_pixels,
                    num_points=occlusion.vehicle_min_num_lidar_points,
                ),
            ],
            lead_config,
        )
        np.testing.assert_allclose(built["observability"][:, row, col], 1.0)

    def test_a_barely_seen_actor_supervises_a_lower_value(
        self,
        lead_config: LeadConfig,
    ) -> None:
        occlusion = lead_config.expert.occlusion
        row, col = cell_of(lead_config, 20.0, 0.0)
        built = build_observability_targets(
            [
                make_box(
                    (20.0, 0.0),
                    visible_pixels=1,
                    num_points=occlusion.vehicle_min_num_lidar_points,
                ),
            ],
            lead_config,
        )
        camera = built["observability"][ObservabilityChannel.CAMERA, row, col]
        lidar = built["observability"][ObservabilityChannel.LIDAR, row, col]
        assert 0.0 < camera < 1.0
        assert lidar == pytest.approx(1.0)

    def test_an_occluded_actor_supervises_zero_not_nothing(
        self,
        lead_config: LeadConfig,
    ) -> None:
        # Zero returns is a measurement, and the informative one: the whole
        # point is that the student learns where its sensors see nothing.
        row, col = cell_of(lead_config, 20.0, 0.0)
        built = build_observability_targets(
            [make_box((20.0, 0.0), visible_pixels=0, num_points=0)],
            lead_config,
        )
        assert built["observability_mask"][:, row, col].all()
        np.testing.assert_allclose(built["observability"][:, row, col], 0.0)

    def test_an_unmeasured_modality_is_left_unsupervised(
        self,
        lead_config: LeadConfig,
    ) -> None:
        row, col = cell_of(lead_config, 20.0, 0.0)
        built = build_observability_targets(
            [make_box((20.0, 0.0), visible_pixels=-1, num_points=7)],
            lead_config,
        )
        mask = built["observability_mask"]
        assert mask[ObservabilityChannel.CAMERA, row, col] == 0.0
        assert mask[ObservabilityChannel.LIDAR, row, col] == 1.0

    def test_the_best_resolved_box_wins_a_shared_cell(
        self,
        lead_config: LeadConfig,
    ) -> None:
        row, col = cell_of(lead_config, 20.0, 0.0)
        built = build_observability_targets(
            [
                make_box(
                    (20.0, 0.0), visible_pixels=0, num_points=0, extent=(0.4, 0.4)
                ),
                make_box(
                    (20.0, 0.0),
                    visible_pixels=500,
                    num_points=500,
                    extent=(0.4, 0.4),
                ),
            ],
            lead_config,
        )
        np.testing.assert_allclose(built["observability"][:, row, col], 1.0)

    def test_pedestrians_use_the_pedestrian_threshold(
        self,
        lead_config: LeadConfig,
    ) -> None:
        occlusion = lead_config.expert.occlusion
        assert (
            occlusion.pedestrian_min_num_visible_pixels
            > occlusion.vehicle_min_num_visible_pixels
        )
        row, col = cell_of(lead_config, 20.0, 0.0)
        pixels = occlusion.vehicle_min_num_visible_pixels

        as_walker = build_observability_targets(
            [make_box((20.0, 0.0), box_class="walker", visible_pixels=pixels)],
            lead_config,
        )
        as_car = build_observability_targets(
            [make_box((20.0, 0.0), box_class="car", visible_pixels=pixels)],
            lead_config,
        )
        channel = ObservabilityChannel.CAMERA
        assert as_walker["observability"][channel, row, col] < 1.0
        assert as_car["observability"][channel, row, col] == pytest.approx(1.0)

    def test_signs_and_lights_carry_no_measurement(
        self,
        lead_config: LeadConfig,
    ) -> None:
        for box_class in ("traffic_light", "stop_sign", "traffic_sign"):
            built = build_observability_targets(
                [make_box((20.0, 0.0), box_class=box_class)],
                lead_config,
            )
            assert built["observability_mask"].sum() == 0.0, box_class

    def test_hard_targets_reproduce_the_experts_decision(
        self,
        lead_config: LeadConfig,
    ) -> None:
        lead_config.policy.transfuser.observability_soft_targets = False
        occlusion = lead_config.expert.occlusion
        row, col = cell_of(lead_config, 20.0, 0.0)

        below = build_observability_targets(
            [
                make_box(
                    (20.0, 0.0),
                    visible_pixels=occlusion.vehicle_min_num_visible_pixels - 1,
                ),
            ],
            lead_config,
        )
        at = build_observability_targets(
            [
                make_box(
                    (20.0, 0.0),
                    visible_pixels=occlusion.vehicle_min_num_visible_pixels,
                ),
            ],
            lead_config,
        )
        channel = ObservabilityChannel.CAMERA
        assert below["observability"][channel, row, col] == 0.0
        assert at["observability"][channel, row, col] == 1.0


class TestObservabilityDecoder:
    """Tests for the head and its masked loss."""

    @pytest.fixture
    def decoder(self, lead_config: LeadConfig) -> ObservabilityDecoder:
        lead_config.policy.transfuser.use_observability = True
        return ObservabilityDecoder(lead_config)

    def test_predicts_one_logit_per_cell_per_modality(
        self,
        lead_config: LeadConfig,
        decoder: ObservabilityDecoder,
    ) -> None:
        config = lead_config.policy.transfuser
        features = torch.randn(2, config.bev_feature_channels, 20, 24)

        prediction = decoder(features)

        assert prediction.shape == (
            2,
            NUM_OBSERVABILITY_CHANNELS,
            config.lidar_height_pixel // config.bev_downsample_factor,
            config.lidar_width_pixel // config.bev_downsample_factor,
        )

    def test_loss_ignores_unsupervised_cells(
        self,
        lead_config: LeadConfig,
        decoder: ObservabilityDecoder,
    ) -> None:
        config = lead_config.policy.transfuser
        shape = (
            1,
            NUM_OBSERVABILITY_CHANNELS,
            config.lidar_height_pixel // config.bev_downsample_factor,
            config.lidar_width_pixel // config.bev_downsample_factor,
        )
        # Perfect where supervised, arbitrary everywhere else.
        target = torch.zeros(shape)
        mask = torch.zeros(shape)
        mask[0, :, 5, 5] = 1.0
        prediction = torch.full(shape, 20.0)
        prediction[0, :, 5, 5] = -20.0

        losses: dict = {}
        decoder.compute_loss(
            prediction,
            {"observability": target, "observability_mask": mask},
            losses,
            {},
        )

        assert losses["loss_observability"].item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_finite_and_differentiable(
        self,
        lead_config: LeadConfig,
        decoder: ObservabilityDecoder,
    ) -> None:
        config = lead_config.policy.transfuser
        features = torch.randn(
            2,
            config.bev_feature_channels,
            20,
            24,
            requires_grad=True,
        )
        prediction = decoder(features)
        target = torch.rand_like(prediction)
        mask = (torch.rand_like(prediction) > 0.5).float()

        losses: dict = {}
        decoder.compute_loss(
            prediction,
            {"observability": target, "observability_mask": mask},
            losses,
            {},
        )
        losses["loss_observability"].backward()

        assert torch.isfinite(losses["loss_observability"])
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_no_loss_when_the_head_is_off(self, lead_config: LeadConfig) -> None:
        lead_config.policy.transfuser.use_observability = False
        decoder = ObservabilityDecoder(lead_config)
        losses: dict = {}

        decoder.compute_loss(torch.zeros(1, 2, 4, 4), {}, losses, {})

        assert losses == {}


class TestLossWeights:
    """Tests for how the task weight follows the head toggle."""

    def test_weight_is_zero_while_the_head_is_off(
        self,
        lead_config: LeadConfig,
    ) -> None:
        lead_config.policy.transfuser.use_observability = False
        weights = lead_config.policy.transfuser.per_task_loss_weights(0)
        assert weights["loss_observability"] == 0.0

    def test_weight_is_nonzero_once_the_head_is_on(
        self,
        lead_config: LeadConfig,
    ) -> None:
        lead_config.policy.transfuser.use_observability = True
        weights = lead_config.policy.transfuser.per_task_loss_weights(0)
        assert weights["loss_observability"] > 0.0
