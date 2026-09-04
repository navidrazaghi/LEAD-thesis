"""Throughput variant: fusion transformers with deformable attention.

The base fusion transformer scores every image anchor against every BEV anchor
and against every other image anchor, which is quadratic in the combined token
count. This variant keeps the fusion structure — same anchor grids, same token
order, same residual back into the branch features — and swaps only the
attention operator for the sparse multi-scale deformable one, so each query
reads a fixed number of learned points per modality instead of the whole
sequence.

Select it with::

    policy.transfuser.backbone_target=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

Where a query reads on the *other* modality is set by
``deformable_calibrated_reference``. Off, it starts at that grid's centre and
has to learn the image/BEV correspondence from scratch. On, it starts where the
rig's calibration says the same piece of world is — see
:mod:`lead.policy.transfuser.encoder.fusion_geometry` — and only refines from
there. The two settings are the ablation isolating what the geometric prior is
worth.

The fusion semantics change: this is a different operator, not a repacking of
the same arithmetic, so it is a model change to be trained and evaluated, not
a drop-in speedup for existing checkpoints.
"""

import jaxtyping as jt
import torch
from torch import nn

from lead.config import LeadConfig
from lead.policy.transfuser.encoder import fusion_geometry
from lead.policy.transfuser.encoder.deformable_attention import (
    MultiScaleDeformableAttention,
    default_reference_points,
)
from lead.policy.transfuser.encoder.observability_gate import ObservabilityGate
from lead.policy.transfuser.encoder.residual_gain import ResidualGain
from lead.policy.transfuser.encoder.transfuser_backbone import (
    GPT,
    TransfuserBackbone,
)


class DeformableBlock(nn.Module):
    """Transformer block whose attention samples instead of scoring all pairs."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_exp: int,
        attn_pdrop: float,
        resid_pdrop: float,
        spatial_shapes: tuple[tuple[int, int], ...],
        num_points: int,
        learn_cross_reference: bool,
        base_reference_points: torch.Tensor | None,
        gated: bool,
        gained: bool,
    ) -> None:
        """Initialize a transformer block with deformable attention.

        Args:
            n_embd: Embedding dimension (feature channels).
            n_head: Number of attention heads.
            block_exp: Expansion factor for MLP hidden dimension.
            attn_pdrop: Dropout probability for the sampled-point weights.
            resid_pdrop: Dropout probability for residual connections.
            spatial_shapes: ``(height, width)`` of each modality's token grid.
            num_points: Sampled points per query, per head, per modality.
            learn_cross_reference: Whether queries refine their reference points.
            base_reference_points: Reference points to start from, or None for
                the operator's geometry-free defaults.
            gated: Whether an observability gate shifts the modality weights.
            gained: Whether a residual gain scales how much of the attention
                output enters the token.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.gate = ObservabilityGate(n_embd, len(spatial_shapes)) if gated else None
        self.residual_gain = ResidualGain(n_embd) if gained else None
        self.attn = MultiScaleDeformableAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            spatial_shapes=spatial_shapes,
            num_points=num_points,
            learn_cross_reference=learn_cross_reference,
            base_reference_points=base_reference_points,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> tuple[
        jt.Float[torch.Tensor, "B T C"],
        jt.Float[torch.Tensor, "B T L"] | None,
    ]:
        """Apply the block with pre-normalization and residual connections.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_embd).

        Returns:
            The output tensor, same shape as the input, and this block's gate
            logits so they can be supervised; None when the block is ungated.
        """
        normalized = self.ln1(x)
        gate_logits = self.gate(normalized) if self.gate is not None else None
        attended = self.attn(normalized, gate_logits)
        if self.residual_gain is not None:
            attended = attended * self.residual_gain(normalized)
        x = x + attended
        return x + self.mlp(self.ln2(x)), gate_logits


class DeformableBlockStack(nn.Module):
    """Runs the blocks in order and keeps the gate logits they produced.

    Stands in for the base transformer's ``nn.Sequential`` so
    :meth:`~lead.policy.transfuser.encoder.transfuser_backbone.GPT.forward` and
    its token packing carry over unchanged, while the per-block gate logits
    still reach the loss.
    """

    def __init__(self, blocks: list[DeformableBlock]) -> None:
        """Hold the blocks of one fusion transformer.

        Args:
            blocks: The blocks, in the order they run.
        """
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.gate_logits: list[torch.Tensor] = []

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T C"]:
        """Run every block, recording the gate logits along the way.

        Args:
            x: The fusion tokens.

        Returns:
            The tokens after every block has run.
        """
        gate_logits: list[torch.Tensor] = []
        for block in self.blocks:
            x, logits = block(x)
            if logits is not None:
                gate_logits.append(logits)
        self.gate_logits = gate_logits
        return x


class DeformableGPT(GPT):
    """The base fusion transformer with its blocks swapped for deformable ones.

    The token layout the base builds — image anchors then BEV anchors, with one
    learned positional embedding over both — is exactly what the deformable
    operator needs, so ``GPT.forward`` is inherited unchanged and the two grids
    become the two levels the attention samples from.
    """

    def __init__(self, n_embd: int, lead_config: LeadConfig) -> None:
        """Build the base GPT, then replace its transformer blocks.

        Args:
            n_embd: Embedding dimension (number of feature channels).
            lead_config: Root config tree.
        """
        super().__init__(n_embd, lead_config)
        config = lead_config.policy.transfuser
        spatial_shapes = (
            (config.img_vert_anchors, config.img_horz_anchors),
            (config.lidar_bev_grid_rows, config.lidar_bev_grid_cols),
        )

        base_reference_points = None
        calibrated_tokens = None
        if config.deformable_calibrated_reference:
            # Every block shares one table: the grids and the rig are fixed, so
            # the correspondence varies neither by layer nor by sample.
            base_reference_points, calibrated_tokens = (
                fusion_geometry.calibrated_reference_points(
                    lead_config,
                    default_reference_points(spatial_shapes),
                    config.deformable_reference_height_meter,
                )
            )
        # Which tokens got a calibrated cross-modal reference, for reporting;
        # None when the geometry-free defaults are in use.
        self.register_buffer("calibrated_tokens", calibrated_tokens, persistent=False)

        self.blocks = DeformableBlockStack(
            [
                DeformableBlock(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                    spatial_shapes,
                    config.deformable_num_points,
                    config.deformable_learn_cross_reference,
                    base_reference_points,
                    config.use_observability_gate,
                    config.use_residual_gain,
                )
                for _ in range(config.n_layer)
            ],
        )
        # The generic init runs first and would otherwise overwrite the offset
        # fan, the zeroed weight matrices the deformable operator needs, and the
        # zeroed gate that keeps a gated model starting where the ungated one does.
        self.apply(self._init_weights)
        for block in self.blocks.blocks:
            block.attn.reset_parameters()
            if block.residual_gain is not None:
                block.residual_gain.reset_parameters()
            if block.gate is not None:
                block.gate.reset_parameters()


class DeformableFusionBackbone(TransfuserBackbone):
    """TransfuserBackbone whose fusion transformers use deformable attention."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the base backbone, then swap the fusion transformers.

        Args:
            lead_config: Root config tree.
        """
        super().__init__(lead_config)
        image_start = 1 if len(self.image_encoder.return_layers) > 4 else 0
        self.transformers = nn.ModuleList(
            [
                DeformableGPT(
                    n_embd=self.image_encoder.feature_info.info[image_start + i][
                        "num_chs"
                    ],
                    lead_config=lead_config,
                )
                for i in range(4)
            ],
        )

    @property
    def gate_logits(self) -> list[torch.Tensor]:
        """The gate logits of every block of every stage of the last forward.

        Read after the backbone has run; empty when the gate is off. The list is
        rebuilt each forward, so a stale read is not possible.

        Returns:
            One tensor per gated block.
        """
        return [
            logits
            for transformer in self.transformers
            for logits in transformer.blocks.gate_logits
        ]
