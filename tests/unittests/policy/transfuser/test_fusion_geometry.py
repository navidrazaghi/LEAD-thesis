"""Tests for the calibration-derived image/BEV token correspondence."""

import math

import numpy as np
import pytest
import torch

from lead.common.geometry import euler_degrees_to_rotation_matrix
from lead.config import LeadConfig, load_lead_config
from lead.policy.transfuser.encoder import fusion_geometry
from lead.policy.transfuser.encoder.deformable_attention import (
    default_reference_points,
)

REFERENCE_HEIGHT = 0.8


@pytest.fixture
def lead_config() -> LeadConfig:
    """Fixture providing the root config tree."""
    return load_lead_config()


@pytest.fixture
def front_camera(lead_config: LeadConfig) -> dict:
    """Fixture providing the forward-facing camera's calibration."""
    return fusion_geometry.stitched_camera_specs(lead_config)[1]


@pytest.fixture
def left_camera(lead_config: LeadConfig) -> dict:
    """Fixture providing the leftmost input camera's calibration."""
    return fusion_geometry.stitched_camera_specs(lead_config)[0]


class TestProjectToCamera:
    """Tests for the ego-frame to pixel projection."""

    def test_optical_axis_lands_on_the_principal_point(
        self,
        left_camera: dict,
    ) -> None:
        yaw = math.radians(left_camera["rot"][2])
        along_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        point = np.array(left_camera["pos"]) + 25.0 * along_axis

        pixels, inside = fusion_geometry.project_to_camera(
            left_camera,
            point[None, :],
        )

        assert inside[0]
        np.testing.assert_allclose(
            pixels[0],
            [left_camera["width"] / 2, left_camera["height"] / 2],
            atol=1e-9,
        )

    def test_rotation_is_the_transpose_of_the_mounting_pose(
        self,
        left_camera: dict,
    ) -> None:
        # The mounting rotation maps camera axes into the ego frame, so taking
        # ego-frame points into the camera needs its transpose. Using it
        # untransposed puts a point on the optical axis behind the camera.
        mounting = euler_degrees_to_rotation_matrix(*left_camera["rot"])
        np.testing.assert_allclose(
            fusion_geometry.world_to_camera_rotation(left_camera["rot"]),
            mounting.T,
            atol=1e-12,
        )

        yaw = math.radians(left_camera["rot"][2])
        offset = 25.0 * np.array([math.cos(yaw), math.sin(yaw), 0.0])
        assert (offset @ mounting.T)[0] < 0.0
        assert (offset @ mounting)[0] > 0.0

    def test_points_behind_the_camera_are_rejected(
        self,
        front_camera: dict,
    ) -> None:
        _, inside = fusion_geometry.project_to_camera(
            front_camera,
            np.array([[-20.0, 0.0, REFERENCE_HEIGHT]]),
        )
        assert not inside[0]

    def test_lateral_offset_moves_the_projection_rightwards(
        self,
        front_camera: dict,
    ) -> None:
        # The ego frame is y-right, so a larger y must give a larger column.
        points = np.array(
            [[30.0, y, REFERENCE_HEIGHT] for y in (-6.0, 0.0, 6.0)],
        )
        pixels, inside = fusion_geometry.project_to_camera(front_camera, points)

        assert inside.all()
        assert pixels[0, 0] < pixels[1, 0] < pixels[2, 0]


class TestGroundPointsOfPixels:
    """Tests for casting pixel rays onto a horizontal plane."""

    def test_inverts_the_projection(self, front_camera: dict) -> None:
        rng = np.random.default_rng(0)
        points = np.stack(
            [
                rng.uniform(5.0, 60.0, 256),
                rng.uniform(-20.0, 20.0, 256),
                np.full(256, REFERENCE_HEIGHT),
            ],
            axis=1,
        )
        pixels, inside = fusion_geometry.project_to_camera(front_camera, points)
        assert inside.any()

        recovered, valid = fusion_geometry.ground_points_of_pixels(
            front_camera,
            pixels[inside],
            REFERENCE_HEIGHT,
        )

        assert valid.all()
        np.testing.assert_allclose(recovered, points[inside, :2], atol=1e-9)

    def test_rays_at_or_above_the_horizon_do_not_land(
        self,
        front_camera: dict,
    ) -> None:
        # The cameras carry no pitch, so the horizon is the principal row.
        horizon = front_camera["height"] / 2
        pixels = np.array([[192.0, horizon], [192.0, horizon - 40.0]])

        _, valid = fusion_geometry.ground_points_of_pixels(
            front_camera,
            pixels,
            REFERENCE_HEIGHT,
        )

        assert not valid.any()


class TestBevCellCentres:
    """Tests for the metric geometry of the BEV token grid."""

    def test_spans_the_configured_crop(self, lead_config: LeadConfig) -> None:
        config = lead_config.policy.transfuser
        centres = fusion_geometry.bev_cell_centres(lead_config)

        assert centres.shape == (
            config.lidar_bev_grid_rows * config.lidar_bev_grid_cols,
            2,
        )
        # Cell centres sit half a cell inside each boundary.
        half_x = (config.bev_max_x_meter - config.bev_min_x_meter) / (
            2 * config.lidar_bev_grid_cols
        )
        half_y = (config.bev_max_y_meter - config.bev_min_y_meter) / (
            2 * config.lidar_bev_grid_rows
        )
        assert centres[:, 0].min() == pytest.approx(config.bev_min_x_meter + half_x)
        assert centres[:, 0].max() == pytest.approx(config.bev_max_x_meter - half_x)
        assert centres[:, 1].min() == pytest.approx(config.bev_min_y_meter + half_y)
        assert centres[:, 1].max() == pytest.approx(config.bev_max_y_meter - half_y)

    def test_flattens_row_major_with_columns_along_x(
        self,
        lead_config: LeadConfig,
    ) -> None:
        config = lead_config.policy.transfuser
        centres = fusion_geometry.bev_cell_centres(lead_config)
        cols = config.lidar_bev_grid_cols

        # Within a row, x advances; across rows at a fixed column, y advances.
        assert centres[0, 0] < centres[1, 0]
        assert centres[0, 1] == pytest.approx(centres[cols - 1, 1])
        assert centres[0, 1] < centres[cols, 1]


class TestBevCellsInImage:
    """Tests for projecting the BEV token grid into the stitched image."""

    def test_nothing_behind_the_ego_is_visible(
        self,
        lead_config: LeadConfig,
    ) -> None:
        centres = fusion_geometry.bev_cell_centres(lead_config)
        _, visible = fusion_geometry.bev_cells_in_image(
            lead_config,
            REFERENCE_HEIGHT,
        )
        assert not visible[centres[:, 0] < -2.0].any()

    def test_some_cells_ahead_are_visible(self, lead_config: LeadConfig) -> None:
        _, visible = fusion_geometry.bev_cells_in_image(
            lead_config,
            REFERENCE_HEIGHT,
        )
        assert visible.any()

    def test_positions_are_normalized_into_the_stitch(
        self,
        lead_config: LeadConfig,
    ) -> None:
        positions, visible = fusion_geometry.bev_cells_in_image(
            lead_config,
            REFERENCE_HEIGHT,
        )
        seen = positions[visible]
        assert (seen >= 0.0).all()
        assert (seen <= 1.0).all()

    def test_the_middle_camera_is_the_middle_of_the_stitch(
        self,
        lead_config: LeadConfig,
        front_camera: dict,
    ) -> None:
        config = lead_config.policy.transfuser
        pixels, inside = fusion_geometry.project_to_camera(
            front_camera,
            np.array([[30.0, 0.0, REFERENCE_HEIGHT]]),
        )
        assert inside[0]
        stitched = (pixels[0, 0] + front_camera["width"]) / config.final_image_width
        assert stitched == pytest.approx(0.5, abs=2e-3)


class TestImageTokensInBev:
    """Tests for casting the image token grid onto the ground plane."""

    def test_validity_grows_towards_the_bottom_of_the_image(
        self,
        lead_config: LeadConfig,
    ) -> None:
        config = lead_config.policy.transfuser
        _, valid = fusion_geometry.image_tokens_in_bev(
            lead_config,
            REFERENCE_HEIGHT,
        )
        per_row = valid.reshape(
            config.img_vert_anchors,
            config.img_horz_anchors,
        ).sum(axis=1)

        assert per_row[0] == 0
        assert per_row[-1] > 0
        assert (np.diff(per_row) >= 0).all()

    def test_the_bottom_centre_token_maps_just_ahead_of_the_ego(
        self,
        lead_config: LeadConfig,
    ) -> None:
        config = lead_config.policy.transfuser
        positions, valid = fusion_geometry.image_tokens_in_bev(
            lead_config,
            REFERENCE_HEIGHT,
        )
        token = (config.img_vert_anchors - 1) * config.img_horz_anchors
        token += config.img_horz_anchors // 2

        assert valid[token]
        x_meter = (
            positions[token, 0] * (config.bev_max_x_meter - config.bev_min_x_meter)
            + config.bev_min_x_meter
        )
        y_meter = (
            positions[token, 1] * (config.bev_max_y_meter - config.bev_min_y_meter)
            + config.bev_min_y_meter
        )
        assert x_meter > 0.0
        assert abs(y_meter) < 2.0


class TestCalibratedReferencePoints:
    """Tests for the assembled reference table the operator consumes."""

    @pytest.fixture
    def built(self, lead_config: LeadConfig) -> tuple:
        config = lead_config.policy.transfuser
        shapes = (
            (config.img_vert_anchors, config.img_horz_anchors),
            (config.lidar_bev_grid_rows, config.lidar_bev_grid_cols),
        )
        defaults = default_reference_points(shapes)
        reference, valid = fusion_geometry.calibrated_reference_points(
            lead_config,
            defaults,
            REFERENCE_HEIGHT,
        )
        return defaults, reference, valid, config

    def test_shape_matches_the_defaults(self, built: tuple) -> None:
        defaults, reference, valid, _ = built
        assert reference.shape == defaults.shape
        assert valid.shape == (defaults.shape[1],)

    def test_own_modality_references_are_untouched(self, built: tuple) -> None:
        defaults, reference, _, config = built
        split = config.img_vert_anchors * config.img_horz_anchors
        torch.testing.assert_close(
            reference[0, :split, 0],
            defaults[0, :split, 0],
        )
        torch.testing.assert_close(
            reference[0, split:, 1],
            defaults[0, split:, 1],
        )

    def test_unprojectable_tokens_keep_the_centre_fallback(
        self,
        built: tuple,
    ) -> None:
        _, reference, valid, config = built
        split = config.img_vert_anchors * config.img_horz_anchors
        unseen = (~valid[:split]).nonzero().flatten()

        assert len(unseen) > 0
        torch.testing.assert_close(
            reference[0, unseen, 1],
            torch.full((len(unseen), 2), 0.5),
        )

    def test_calibrated_tokens_actually_move(self, built: tuple) -> None:
        defaults, reference, valid, _ = built
        moved = (reference - defaults).abs().amax(dim=-1)[0].amax(dim=-1)
        assert (moved[valid] > 1e-6).any()

    def test_every_reference_stays_inside_the_grid(self, built: tuple) -> None:
        _, reference, _, _ = built
        assert (reference >= 0.0).all()
        assert (reference <= 1.0).all()
