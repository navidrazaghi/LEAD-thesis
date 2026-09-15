"""Tests for the deformable fusion attention operator."""

import pytest
import torch

from lead.config import load_lead_config
from lead.policy.transfuser.encoder.deformable_attention import (
    MultiScaleDeformableAttention,
    deformable_aggregate,
    inverse_sigmoid,
)
from lead.policy.transfuser.encoder.transfuser_backbone import SelfAttention

IMAGE_SHAPE = (12, 36)
BEV_SHAPE = (10, 12)
SPATIAL_SHAPES = (IMAGE_SHAPE, BEV_SHAPE)
NUM_IMAGE_TOKENS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
NUM_TOKENS = NUM_IMAGE_TOKENS + BEV_SHAPE[0] * BEV_SHAPE[1]


@pytest.fixture
def attention() -> MultiScaleDeformableAttention:
    """Fixture providing a deformable attention module over both anchor grids."""
    torch.manual_seed(0)
    return MultiScaleDeformableAttention(
        n_embd=64,
        n_head=4,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        spatial_shapes=SPATIAL_SHAPES,
        num_points=4,
    )


def _naive_aggregate(
    value: torch.Tensor,
    spatial_shapes: tuple[tuple[int, int], ...],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Bilinear sampling written straight from its definition, for comparison."""
    batch, _, heads, dim = value.shape
    tokens = sampling_locations.shape[1]
    points = sampling_locations.shape[4]
    level_values = value.split([h * w for h, w in spatial_shapes], dim=1)
    out = torch.zeros(batch, tokens, heads, dim, dtype=value.dtype)

    for level, (height, width) in enumerate(spatial_shapes):
        grid = level_values[level].view(batch, height, width, heads, dim)
        for point in range(points):
            # ``align_corners=False`` maps [0, 1] onto pixel centres.
            px = sampling_locations[:, :, :, level, point, 0] * width - 0.5
            py = sampling_locations[:, :, :, level, point, 1] * height - 0.5
            x0, y0 = torch.floor(px), torch.floor(py)
            fx, fy = px - x0, py - y0
            for dy in (0, 1):
                for dx in (0, 1):
                    xi, yi = (x0 + dx).long(), (y0 + dy).long()
                    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
                    corner = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * inside
                    xi, yi = xi.clamp(0, width - 1), yi.clamp(0, height - 1)
                    gathered = grid[
                        torch.arange(batch).view(batch, 1, 1),
                        yi,
                        xi,
                        torch.arange(heads).view(1, 1, heads),
                    ]
                    weight = corner * attention_weights[:, :, :, level, point]
                    out += gathered * weight.unsqueeze(-1)

    return out.reshape(batch, tokens, heads * dim)


class TestDeformableAggregate:
    """Tests for the sampling and weighted-sum kernel."""

    def test_matches_explicit_bilinear_sampling(self) -> None:
        torch.manual_seed(0)
        batch, heads, dim, points = 2, 4, 8, 4
        value = torch.randn(batch, NUM_TOKENS, heads, dim, dtype=torch.float64)
        # Deliberately outside [0, 1] as well, to cover the zero padding.
        locations = (
            torch.rand(batch, NUM_TOKENS, heads, 2, points, 2, dtype=torch.float64)
            * 1.2
            - 0.1
        )
        weights = torch.rand(batch, NUM_TOKENS, heads, 2, points, dtype=torch.float64)
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True)

        result = deformable_aggregate(value, SPATIAL_SHAPES, locations, weights)
        expected = _naive_aggregate(value, SPATIAL_SHAPES, locations, weights)

        torch.testing.assert_close(result, expected)

    def test_samples_outside_the_grid_read_zero(self) -> None:
        value = torch.ones(1, NUM_TOKENS, 2, 4)
        locations = torch.full((1, NUM_TOKENS, 2, 2, 1, 2), 5.0)
        weights = torch.full((1, NUM_TOKENS, 2, 2, 1), 0.5)

        result = deformable_aggregate(value, SPATIAL_SHAPES, locations, weights)

        torch.testing.assert_close(result, torch.zeros_like(result))


class TestInverseSigmoid:
    """Tests for the logit helper used to refine reference points."""

    def test_round_trips_through_sigmoid(self) -> None:
        x = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
        torch.testing.assert_close(torch.sigmoid(inverse_sigmoid(x)), x)

    def test_clamps_the_saturating_ends(self) -> None:
        result = inverse_sigmoid(torch.tensor([0.0, 1.0]))
        assert torch.isfinite(result).all()


class TestReferencePoints:
    """Tests for the per-modality reference geometry."""

    def test_own_level_reference_is_the_query_cell_centre(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        base = attention.base_reference_points
        assert base.shape == (1, NUM_TOKENS, 2, 2)
        # First image token: first cell of the 12x36 grid, as (x, y).
        torch.testing.assert_close(
            base[0, 0, 0],
            torch.tensor([0.5 / 36.0, 0.5 / 12.0]),
        )
        # Last BEV token: last cell of the 10x12 grid.
        torch.testing.assert_close(
            base[0, -1, 1],
            torch.tensor([11.5 / 12.0, 9.5 / 10.0]),
        )

    def test_other_level_reference_starts_at_the_grid_centre(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        base = attention.base_reference_points
        centre = torch.tensor([0.5, 0.5])
        # An image query has no geometric position on the BEV grid, and vice versa.
        torch.testing.assert_close(base[0, 0, 1], centre)
        torch.testing.assert_close(base[0, NUM_IMAGE_TOKENS, 0], centre)

    def test_learned_refinement_is_a_no_op_at_initialization(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        x = torch.randn(2, NUM_TOKENS, 64)
        torch.testing.assert_close(
            attention.reference_points(x),
            attention.base_reference_points.expand(2, -1, -1, -1),
        )

    def test_fixed_reference_variant_drops_the_projection(self) -> None:
        attention = MultiScaleDeformableAttention(
            n_embd=64,
            n_head=4,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            spatial_shapes=SPATIAL_SHAPES,
            learn_cross_reference=False,
        )
        assert attention.reference_delta is None
        x = torch.randn(2, NUM_TOKENS, 64)
        assert attention(x).shape == x.shape


class TestMultiScaleDeformableAttention:
    """Tests for the drop-in attention module."""

    def test_preserves_the_self_attention_contract(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        vanilla = SelfAttention(64, 4, 0.1, 0.1)
        x = torch.randn(2, NUM_TOKENS, 64)
        attention.eval()
        vanilla.eval()
        assert attention(x).shape == vanilla(x).shape

    def test_output_is_finite_and_differentiable(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        x = torch.randn(2, NUM_TOKENS, 64, requires_grad=True)
        attention.eval()

        output = attention(x)
        output.sum().backward()

        assert torch.isfinite(output).all()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None for p in attention.parameters())

    def test_initial_attention_weights_are_uniform(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        x = torch.randn(2, NUM_TOKENS, 64)
        assert attention.attention_weights(x).abs().max() == 0.0

    def test_sampling_offsets_fan_out_across_heads(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        bias = attention.sampling_offsets.bias.view(4, 2, 4, 2)
        assert (attention.sampling_offsets.weight == 0.0).all()
        # Point k sits k+1 cells out along the head's own direction.
        radii = bias[:, 0].norm(dim=-1)
        assert torch.all(radii[:, 1:] > radii[:, :-1])

    def test_rejects_a_head_count_that_does_not_divide_the_width(self) -> None:
        with pytest.raises(AssertionError):
            MultiScaleDeformableAttention(
                n_embd=64,
                n_head=5,
                attn_pdrop=0.0,
                resid_pdrop=0.0,
                spatial_shapes=SPATIAL_SHAPES,
            )


class TestConfiguredGeometry:
    """The anchor grids the shipped config produces."""

    def test_matches_the_shapes_the_tests_assume(self) -> None:
        config = load_lead_config().policy.transfuser
        assert (config.img_vert_anchors, config.img_horz_anchors) == IMAGE_SHAPE
        assert (config.lidar_bev_grid_rows, config.lidar_bev_grid_cols) == BEV_SHAPE
