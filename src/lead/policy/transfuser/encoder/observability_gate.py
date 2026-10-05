"""The modality gate of the deformable fusion, and the targets that train it.

The deformable operator already normalizes each query's sampled points over the
``(modality, point)`` axes with one softmax. Which modality a query ends up
reading from is therefore decided by those logits, and shifting the balance
needs no new mechanism — only a per-token, per-modality bias added before the
softmax. That bias is what this module predicts.

It is supervised by the same targets the dense observability head is trained on
(:mod:`lead.policy.transfuser.dataloader.observability`), mapped onto the two
token grids the fusion works with. A BEV token pools the cells it covers. An
image token has no footprint in BEV, so it takes the cells its ray reaches,
through the calibration
:mod:`lead.policy.transfuser.encoder.fusion_geometry` already computes.

The gate starts as a no-op: its projection is zero-initialized, so a gated model
begins exactly where the ungated one does and any difference is learned.
"""

import jaxtyping as jt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.observability import (
    NUM_OBSERVABILITY_CHANNELS,
)
from lead.policy.transfuser.encoder import fusion_geometry

# Coverage below this leaves a token unsupervised rather than dividing by it.
_MIN_COVERAGE = 1e-6

# DeformableFusionBackbone builds spatial_shapes camera-first, then LiDAR,
# and the gate's modality axis is that tuple's order. Reading these two the
# wrong way round would silently mask the intact sensor, so they are named
# here rather than written as literals at the point of use.
_MODALITY_INDEX = {"camera": 0, "lidar": 1}


class ObservabilityGate(nn.Module):
    """Per-token, per-modality bias on the deformable operator's modality logits."""

    def __init__(
        self,
        n_embd: int,
        num_levels: int,
        lead_config: LeadConfig,
    ) -> None:
        """Initialize the gate as a no-op.

        Args:
            n_embd: Embedding dimension of the fusion tokens.
            num_levels: Number of modalities the operator samples from.
            lead_config: Root config tree, read at forward time for the oracle
                substitution. Held as a plain attribute: it is not a module or
                a tensor, so it stays out of the state dict.
        """
        super().__init__()
        self.head = nn.Linear(n_embd, num_levels)
        self.lead_config = lead_config
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Zero the projection so the gate starts neutral.

        Must run after any generic initialization that would overwrite it,
        otherwise the gated model does not start from the ungated one.
        """
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T L"]:
        """Predict how much each query should trust each modality.

        Args:
            x: The fusion tokens, already normalized by the block.

        Returns:
            One logit per token per modality, added to the operator's own
            modality logits before it normalizes them.
        """
        if not self.lead_config.evaluation.inference.oracle_gate:
            return self.head(x)
        return self._oracle_logits(x)

    def _oracle_logits(
        self,
        x: jt.Float[torch.Tensor, "B T C"],
    ) -> jt.Float[torch.Tensor, "B T L"]:
        """The bias a gate with perfect knowledge of the damage would produce.

        The harness applies one modality fault at a known severity, uniform over
        that modality, so the truth is a scalar per modality and the same bias
        applies to every token.

        Args:
            x: The fusion tokens, for shape, device and dtype only.

        Returns:
            Zero for the intact modality, and a negative bias scaled by severity
            for the damaged one.

        Raises:
            ValueError: If a spatial degrade_family is active. Its severity
                varies over the image, so no scalar stands in for it, and
                returning a uniform bias would quietly measure the wrong thing.
        """
        inference = self.lead_config.evaluation.inference
        if inference.degrade_family != "none":
            raise ValueError(
                f"oracle_gate has no ground truth for the spatial family "
                f"'{inference.degrade_family}'; it is defined for "
                f"degrade_modality only.",
            )
        logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.head.out_features,
            device=x.device,
            dtype=x.dtype,
        )
        index = _MODALITY_INDEX.get(inference.degrade_modality)
        if index is None:
            # Nothing is damaged, so the truth is that both are reliable and a
            # neutral bias is the honest oracle rather than a special case.
            return logits
        logits[..., index] = -inference.oracle_gate_strength * inference.degrade_severity
        return logits


class ObservabilityTokenTargets(nn.Module):
    """Maps the dense observability targets onto the fusion token grids.

    Holds no parameters; the image tokens' lookup into the dense grid is fixed
    by the rig and the pooling, so it is precomputed once as a buffer.
    """

    def __init__(self, lead_config: LeadConfig) -> None:
        """Precompute where each image token reads the dense target.

        Args:
            lead_config: Root config tree.
        """
        super().__init__()
        config = lead_config.policy.transfuser
        self.cell_rows = config.lidar_height_pixel // config.bev_downsample_factor
        self.cell_cols = config.lidar_width_pixel // config.bev_downsample_factor
        self.bev_rows = config.lidar_bev_grid_rows
        self.bev_cols = config.lidar_bev_grid_cols

        positions, valid = fusion_geometry.image_tokens_in_bev(
            lead_config,
            config.deformable_reference_height_meter,
        )
        columns = np.clip(
            (positions[:, 0] * self.cell_cols).astype(np.int64),
            0,
            self.cell_cols - 1,
        )
        rows = np.clip(
            (positions[:, 1] * self.cell_rows).astype(np.int64),
            0,
            self.cell_rows - 1,
        )
        # Tokens whose ray never reaches the ground read cell zero and are
        # masked out, so the index only has to be in range.
        self.register_buffer(
            "image_cell_index",
            torch.from_numpy(np.where(valid, rows * self.cell_cols + columns, 0)),
            persistent=False,
        )
        self.register_buffer(
            "image_token_valid",
            torch.from_numpy(valid.astype(np.float32)),
            persistent=False,
        )

    def forward(
        self,
        target: jt.Float[torch.Tensor, "B M cell_h cell_w"],
        mask: jt.Float[torch.Tensor, "B M cell_h cell_w"],
    ) -> tuple[
        jt.Float[torch.Tensor, "B T M"],
        jt.Float[torch.Tensor, "B T M"],
    ]:
        """Reduce the dense targets to one value per fusion token per modality.

        Args:
            target: Dense per-cell observability targets.
            mask: Which cells of ``target`` carry a measurement.

        Returns:
            The per-token targets and their mask, image tokens first, in the
            order the fusion transformer concatenates them.
        """
        batch = target.shape[0]

        # BEV tokens pool their own cells; the ratio of the two pooled sums is
        # the mean over the supervised cells alone.
        pooled_size = (self.bev_rows, self.bev_cols)
        covered = F.adaptive_avg_pool2d(target * mask, pooled_size)
        coverage = F.adaptive_avg_pool2d(mask, pooled_size)
        bev_target = covered / coverage.clamp(min=_MIN_COVERAGE)
        bev_mask = (coverage > _MIN_COVERAGE).to(target.dtype)

        # Image tokens read the cell their ray lands in.
        flat_target = target.flatten(2)
        flat_mask = mask.flatten(2)
        index = self.image_cell_index.view(1, 1, -1).expand(
            batch,
            NUM_OBSERVABILITY_CHANNELS,
            -1,
        )
        image_target = flat_target.gather(2, index)
        image_mask = flat_mask.gather(2, index) * self.image_token_valid.view(1, 1, -1)

        return (
            torch.cat(
                [image_target.transpose(1, 2), bev_target.flatten(2).transpose(1, 2)],
                dim=1,
            ),
            torch.cat(
                [image_mask.transpose(1, 2), bev_mask.flatten(2).transpose(1, 2)],
                dim=1,
            ),
        )


def gate_loss(
    gate_logits: list[jt.Float[torch.Tensor, "B T L"]],
    target: jt.Float[torch.Tensor, "B T L"],
    mask: jt.Float[torch.Tensor, "B T L"],
) -> jt.Float[torch.Tensor, ""]:
    """Masked binary cross entropy of every fusion block's gate.

    Args:
        gate_logits: The gate logits of each block, in forward order.
        target: Per-token observability targets.
        mask: Which token/modality pairs carry a measurement.

    Returns:
        The mean loss over the blocks; zero when nothing is supervised.
    """
    supervised = mask.sum().clamp(min=1.0)
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    for logits in gate_logits:
        elementwise = F.binary_cross_entropy_with_logits(
            logits.float(),
            target,
            reduction="none",
        )
        total = total + (elementwise * mask).sum() / supervised
    return total / max(len(gate_logits), 1)
