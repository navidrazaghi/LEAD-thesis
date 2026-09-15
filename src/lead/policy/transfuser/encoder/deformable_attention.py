"""Multi-scale deformable attention for the TransFuser fusion transformers.

The base fusion transformer in :mod:`lead.policy.transfuser.encoder.transfuser_backbone`
runs dense self-attention over the concatenation of the image anchor grid and
the BEV anchor grid, so each of the ``T = H_img*W_img + H_bev*W_bev`` tokens
scores against all ``T`` of them. This module replaces that with the sparse
operator of Deformable DETR (Zhu et al., ICLR 2021): every query predicts
``num_points`` sampling offsets per modality and reads bilinearly interpolated
values only there, which makes the cost ``O(T * L * K)`` instead of ``O(T^2)``.

Geometry note. In Deformable DETR the levels are one scene at several scales,
so a query's normalized position carries across levels unchanged. Here the two
levels are different modalities — a perspective image grid and a top-down BEV
grid — and no such identity holds: the same normalized coordinate means
different things in each. A query therefore keeps its own grid position as the
reference point on its own modality, and starts at the centre of the other
modality with a learned, content-predicted refinement on top. The network
learns the image/BEV correspondence rather than assuming one.
"""

import math

import jaxtyping as jt
import torch
import torch.nn.functional as F
from torch import nn

# Reference points are refined in logit space, so they are clamped away from
# the saturating ends of the sigmoid before the inverse is taken.
_REFERENCE_EPS = 1e-5


def inverse_sigmoid(
    x: jt.Float[torch.Tensor, "..."],
) -> jt.Float[torch.Tensor, "..."]:
    """Logit of a value in the open unit interval.

    Args:
        x: Tensor with values in ``[0, 1]``.

    Returns:
        The logit of ``x``, clamped away from the saturating ends.
    """
    x = x.clamp(min=_REFERENCE_EPS, max=1.0 - _REFERENCE_EPS)
    return torch.log(x / (1.0 - x))


def deformable_aggregate(
    value: jt.Float[torch.Tensor, "B T n_head head_dim"],
    spatial_shapes: tuple[tuple[int, int], ...],
    sampling_locations: jt.Float[torch.Tensor, "B T n_head L K 2"],
    attention_weights: jt.Float[torch.Tensor, "B T n_head L K"],
) -> jt.Float[torch.Tensor, "B T C"]:
    """Sample each level bilinearly at the given locations and take the weighted sum.

    The pure-PyTorch form of the Deformable DETR kernel: no custom CUDA
    extension to build, and it traces cleanly under ``torch.compile``.

    Args:
        value: Per-head values for every token, ordered level after level.
        spatial_shapes: ``(height, width)`` of each level, in the order the
            tokens of ``value`` are concatenated.
        sampling_locations: Sampling positions in ``[0, 1]`` as ``(x, y)``,
            where ``x`` indexes width and ``y`` indexes height.
        attention_weights: Weight of each sampled point, already normalized
            over the ``(level, point)`` axes.

    Returns:
        The aggregated features, with the heads concatenated back into ``C``.
    """
    batch_size, _, num_heads, head_dim = value.shape
    num_tokens = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]

    # grid_sample wants the grid in [-1, 1] and in the same dtype as the input.
    sampling_grids = (2.0 * sampling_locations - 1.0).to(value.dtype)
    level_values = value.split([h * w for h, w in spatial_shapes], dim=1)

    sampled_per_level = []
    for level, (height, width) in enumerate(spatial_shapes):
        # (B, H*W, n_head, head_dim) -> (B * n_head, head_dim, H, W)
        level_value = (
            level_values[level]
            .permute(0, 2, 3, 1)
            .reshape(batch_size * num_heads, head_dim, height, width)
        )
        # (B, T, n_head, K, 2) -> (B * n_head, T, K, 2)
        level_grid = (
            sampling_grids[:, :, :, level]
            .transpose(1, 2)
            .reshape(batch_size * num_heads, num_tokens, num_points, 2)
        )
        # One read per query per point, shaped (B * n_head, head_dim, T, K).
        sampled_per_level.append(
            F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ),
        )

    # The levels stack into (B * n_head, head_dim, T, L, K) to be weighted.
    sampled = torch.stack(sampled_per_level, dim=-2)
    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_heads,
        1,
        num_tokens,
        len(spatial_shapes),
        num_points,
    )
    output = (sampled * weights).sum(dim=(-2, -1))
    return (
        output.view(batch_size, num_heads * head_dim, num_tokens)
        .transpose(1, 2)
        .contiguous()
    )


def default_reference_points(
    spatial_shapes: tuple[tuple[int, int], ...],
) -> jt.Float[torch.Tensor, "1 T L 2"]:
    """Reference point of every query on every level, knowing nothing but the grids.

    On its own level a query sits at its own cell centre. On every other level
    it starts at the centre of that grid, because without a calibration a
    perspective coordinate and a BEV coordinate are not the same place. See
    :mod:`lead.policy.transfuser.encoder.fusion_geometry` for the table that
    replaces those centres when the rig geometry is used.

    Args:
        spatial_shapes: ``(height, width)`` of each level.

    Returns:
        Reference points in ``[0, 1]`` as ``(x, y)``.
    """
    num_levels = len(spatial_shapes)
    per_level = []
    for height, width in spatial_shapes:
        y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        own = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        # Start at the centre of every level, then overwrite the own level.
        level_points = own.new_full((own.shape[0], num_levels, 2), 0.5)
        per_level.append((own, level_points))

    for level, (own, level_points) in enumerate(per_level):
        level_points[:, level, :] = own

    return torch.cat([points for _, points in per_level], dim=0).unsqueeze(0)


class MultiScaleDeformableAttention(nn.Module):
    """Sparse replacement for the fusion transformer's dense self-attention.

    Drop-in for
    :class:`~lead.policy.transfuser.encoder.transfuser_backbone.SelfAttention`:
    same ``(B, T, C) -> (B, T, C)`` contract, same token order (image grid
    first, BEV grid second). The token grids are fixed by the backbone's
    adaptive pooling, so the reference points are precomputed once here rather
    than passed in per call.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        spatial_shapes: tuple[tuple[int, int], ...],
        num_points: int = 4,
        learn_cross_reference: bool = True,
        base_reference_points: jt.Float[torch.Tensor, "1 T L 2"] | None = None,
    ) -> None:
        """Initialize multi-scale deformable attention over fixed token grids.

        Args:
            n_embd: Embedding dimension (must be divisible by ``n_head``).
            n_head: Number of attention heads.
            attn_pdrop: Dropout probability for the sampled-point weights.
            resid_pdrop: Dropout probability for the output projection.
            spatial_shapes: ``(height, width)`` of each modality's token grid,
                in the order the tokens are concatenated.
            num_points: Sampled points per query, per head, per modality.
            learn_cross_reference: If true, each query refines its reference
                point from its own content; if false the reference points stay
                wherever they are seeded.
            base_reference_points: Reference points to start from, replacing the
                geometry-free defaults. This is where a calibration-derived
                image/BEV correspondence enters — see
                :mod:`lead.policy.transfuser.encoder.fusion_geometry`.

        Raises:
            AssertionError: If ``n_embd`` is not divisible by ``n_head``.
        """
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.spatial_shapes = spatial_shapes
        self.num_levels = len(spatial_shapes)
        self.num_points = num_points
        self.learn_cross_reference = learn_cross_reference

        self.value_proj = nn.Linear(n_embd, n_embd)
        self.sampling_offsets = nn.Linear(
            n_embd,
            n_head * self.num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            n_embd,
            n_head * self.num_levels * num_points,
        )
        self.reference_delta = (
            nn.Linear(n_embd, self.num_levels * 2) if learn_cross_reference else None
        )
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        if base_reference_points is None:
            base_reference_points = default_reference_points(spatial_shapes)
        self.register_buffer(
            "base_reference_points",
            base_reference_points.detach().clone(),
            persistent=False,
        )
        offset_normalizer = torch.tensor(
            [[float(width), float(height)] for height, width in spatial_shapes],
        )
        self.register_buffer("offset_normalizer", offset_normalizer, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the Deformable DETR initialization.

        The offset biases fan the initial sampling points out along
        ``n_head`` evenly spaced directions at increasing radii, so the heads
        start by reading different neighbourhoods; the offset and weight
        matrices start at zero, so at step zero every query reads a uniform
        average of that fan. Must run after any generic weight initialization
        that would otherwise overwrite it.
        """
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_head, dtype=torch.float32) * (
            2.0 * math.pi / self.n_head
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_head, 1, 1, 2).repeat(
            1,
            self.num_levels,
            self.num_points,
            1,
        )
        for point in range(self.num_points):
            grid_init[:, :, point, :] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        if self.reference_delta is not None:
            # Zero start: the reference points equal the geometric defaults.
            nn.init.constant_(self.reference_delta.weight, 0.0)
            nn.init.constant_(self.reference_delta.bias, 0.0)

        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def reference_points(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T L 2"]:
        """Reference point of every query on every level.

        Args:
            x: Input tokens.

        Returns:
            Reference points in ``[0, 1]`` as ``(x, y)``.
        """
        base = self.base_reference_points
        if self.reference_delta is None:
            return base.expand(x.shape[0], -1, -1, -1)
        delta = self.reference_delta(x).view(
            x.shape[0],
            x.shape[1],
            self.num_levels,
            2,
        )
        return torch.sigmoid(inverse_sigmoid(base) + delta.float())

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
        modality_bias: jt.Float[torch.Tensor, "B T L"] | None = None,
    ) -> jt.Float[torch.Tensor, "B T C"]:
        """Compute deformable attention over the concatenated modality grids.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_embd), with the
                image grid's tokens first and the BEV grid's tokens second.
            modality_bias: Per-token, per-modality shift of the sampling
                weights, applied in logit space before they are normalized.
                This is where an observability gate enters; None leaves the
                operator's own logits untouched.

        Returns:
            Attention output tensor of shape (batch, sequence_length, n_embd).
        """
        b, t, c = x.size()

        value = self.value_proj(x).view(b, t, self.n_head, self.head_dim)
        offsets = self.sampling_offsets(x).view(
            b,
            t,
            self.n_head,
            self.num_levels,
            self.num_points,
            2,
        )
        logits = self.attention_weights(x).view(
            b,
            t,
            self.n_head,
            self.num_levels,
            self.num_points,
        )
        if modality_bias is not None:
            # One shift per modality, shared by that modality's heads and
            # points: it says how much to read there, not where.
            logits = logits + modality_bias[:, :, None, :, None]
        weights = F.softmax(logits.flatten(-2), dim=-1).view(
            b,
            t,
            self.n_head,
            self.num_levels,
            self.num_points,
        )
        weights = self.attn_drop(weights)

        # Offsets are predicted in grid cells, so they are normalized by the
        # level's own resolution before being added to the reference point.
        reference = self.reference_points(x)
        sampling_locations = (
            reference[:, :, None, :, None, :]
            + offsets / self.offset_normalizer[None, None, None, :, None, :]
        )

        y = deformable_aggregate(
            value,
            self.spatial_shapes,
            sampling_locations,
            weights,
        )
        return self.resid_drop(self.proj(y))
