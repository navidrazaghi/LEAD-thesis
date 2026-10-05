"""The camera-to-LiDAR hallucination head.

What has to hold: the head only predicts where a camera actually looks, the
supervision does not reach back into the LiDAR branch, the substitution is
proportional to the damage the harness applied, and with the flag off nothing
about the model changes. The last one matters most while evaluations are queued
against the same tree.
"""

import pytest
import torch

from lead.config import load_lead_config
from lead.policy.transfuser.encoder import fusion_geometry
from lead.policy.transfuser.encoder.cross_modal_hallucination import (
    CrossModalHallucination,
    blend,
    hallucination_loss,
    lidar_reliability,
)

ROWS, COLS = 10, 12
IMAGE_CHANNELS, LIDAR_CHANNELS = 16, 8


@pytest.fixture
def config():
    """A default config tree, with nothing damaged."""
    return load_lead_config()


@pytest.fixture
def head(config):
    """A hallucination head sized for the small grids used here."""
    return CrossModalHallucination(config, IMAGE_CHANNELS, LIDAR_CHANNELS)


@pytest.fixture
def image_features():
    """One small batch of camera features."""
    torch.manual_seed(0)
    return torch.randn(2, IMAGE_CHANNELS, 22, 78)


class TestGeometry:
    """The correspondence the head samples through."""

    def test_default_grid_is_unchanged(self, config):
        """Naming the token grid explicitly matches asking for the default."""
        height = config.policy.transfuser.deformable_reference_height_meter
        implicit = fusion_geometry.bev_cells_in_image(config, height)
        explicit = fusion_geometry.bev_cells_in_image(
            config,
            height,
            config.policy.transfuser.lidar_bev_grid_rows,
            config.policy.transfuser.lidar_bev_grid_cols,
        )
        assert (implicit[0] == explicit[0]).all()
        assert (implicit[1] == explicit[1]).all()

    @pytest.mark.parametrize(("rows", "cols"), [(10, 12), (20, 24), (40, 48)])
    def test_visible_fraction_is_geometry_not_resolution(self, config, rows, cols):
        """Which part of the BEV a camera covers cannot depend on the grid."""
        height = config.policy.transfuser.deformable_reference_height_meter
        _, visible = fusion_geometry.bev_cells_in_image(config, height, rows, cols)
        assert 0.6 < visible.mean() < 0.72

    def test_the_blind_part_is_behind_the_ego(self, config):
        """The cameras face forward, so the cells with no view are the rear ones."""
        height = config.policy.transfuser.deformable_reference_height_meter
        _, visible = fusion_geometry.bev_cells_in_image(config, height, ROWS, COLS)
        centres = fusion_geometry.bev_cell_centres(config, ROWS, COLS)
        assert centres[visible, 0].min() > centres[~visible, 0].max()


class TestHead:
    """What the head produces."""

    def test_predicts_the_lidar_grid_shape(self, head, image_features):
        predicted, mask = head(image_features, ROWS, COLS)
        assert predicted.shape == (2, LIDAR_CHANNELS, ROWS, COLS)
        assert mask.shape == (1, 1, ROWS, COLS)

    def test_is_silent_where_no_camera_looks(self, head, image_features):
        """Cells outside every field of view are left at zero, not invented."""
        predicted, mask = head(image_features, ROWS, COLS)
        assert torch.all(predicted[:, :, mask[0, 0] == 0] == 0.0)
        assert torch.any(mask == 0.0), "this rig should have blind cells"

    def test_serves_any_level_resolution(self, head, image_features):
        """The fusion levels carry the same extent at different resolutions."""
        for rows, cols in ((10, 12), (20, 24), (40, 48)):
            predicted, _ = head(image_features, rows, cols)
            assert predicted.shape == (2, LIDAR_CHANNELS, rows, cols)

    def test_correspondence_is_cached_per_shape(self, head, image_features):
        head(image_features, ROWS, COLS)
        head(image_features, ROWS, COLS)
        head(image_features, 20, 24)
        assert sorted(head._correspondence_cache) == [(10, 12), (20, 24)]


class TestLoss:
    """How the head is supervised."""

    def test_does_not_pull_on_the_lidar_branch(self, head, image_features):
        """The target is detached: the camera learns the LiDAR, not the reverse."""
        predicted, mask = head(image_features, ROWS, COLS)
        target = torch.randn(2, LIDAR_CHANNELS, ROWS, COLS, requires_grad=True)

        hallucination_loss(predicted, target, mask).backward()

        assert target.grad is None

    def test_ignores_the_cells_with_no_supervision(self, head, image_features):
        """A wild difference on a blind cell must not enter the loss."""
        predicted, mask = head(image_features, ROWS, COLS)
        target = torch.zeros(2, LIDAR_CHANNELS, ROWS, COLS)
        blind = (mask[0, 0] == 0).nonzero()[0]

        before = hallucination_loss(predicted, target, mask)
        target[:, :, blind[0], blind[1]] = 1e6
        after = hallucination_loss(predicted, target, mask)

        torch.testing.assert_close(before, after)


class TestSubstitution:
    """What happens at inference."""

    def test_reliability_reads_the_applied_damage(self, config):
        inference = config.evaluation.inference
        assert lidar_reliability(config) == 1.0

        inference.degrade_modality = "lidar"
        inference.degrade_severity = 1.0
        assert lidar_reliability(config) == 0.0

        inference.degrade_severity = 0.25
        assert lidar_reliability(config) == pytest.approx(0.75)

    def test_a_damaged_camera_leaves_the_lidar_alone(self, config):
        """This head replaces LiDAR; camera damage is not its business."""
        inference = config.evaluation.inference
        inference.degrade_modality = "camera"
        inference.degrade_severity = 1.0
        assert lidar_reliability(config) == 1.0

    def test_refuses_a_spatial_family(self, config):
        config.evaluation.inference.degrade_family = "occlusion"
        with pytest.raises(ValueError, match="no scalar reliability"):
            lidar_reliability(config)

    def test_intact_lidar_is_untouched(self, head, image_features):
        predicted, mask = head(image_features, ROWS, COLS)
        lidar = torch.randn(2, LIDAR_CHANNELS, ROWS, COLS)

        torch.testing.assert_close(blend(lidar, predicted, mask, 1.0), lidar)

    def test_destroyed_lidar_is_replaced_only_where_seen(
        self,
        head,
        image_features,
    ):
        predicted, mask = head(image_features, ROWS, COLS)
        lidar = torch.randn(2, LIDAR_CHANNELS, ROWS, COLS)

        blended = blend(lidar, predicted, mask, 0.0)

        seen = mask[0, 0] == 1
        torch.testing.assert_close(blended[:, :, seen], predicted[:, :, seen])
        torch.testing.assert_close(blended[:, :, ~seen], lidar[:, :, ~seen])


class TestDefaultOff:
    """The flag is off, and off has to mean nothing changed."""

    def test_flags_default_to_off(self, config):
        assert config.policy.transfuser.use_cross_modal_hallucination is False
        assert config.evaluation.inference.hallucinate_missing_lidar is False

    def test_no_head_is_built(self, config):
        from lead.policy.transfuser.encoder.transfuser_backbone import (
            TransfuserBackbone,
        )

        backbone = TransfuserBackbone(config)
        assert backbone.cross_modal_hallucination is None
        assert backbone.hallucination is None
