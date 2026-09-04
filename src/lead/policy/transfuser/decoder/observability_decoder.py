"""The head predicting how well each modality resolves each BEV cell.

Reads the same top-down BEV feature grid the box and BEV-semantic heads read,
and predicts one logit per cell per modality. Supervision is sparse — only
cells the expert actually measured an actor in carry a target — so the loss is
masked.

The prediction is the quantity the gated fusion consumes: it says, per place in
the world, how much of the evidence there is camera evidence and how much is
LiDAR evidence.
"""

import jaxtyping as jt
import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F

from lead.api.abstract_policy import AuxiliaryLog, TaskLosses
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.observability import (
    NUM_OBSERVABILITY_CHANNELS,
    ObservabilityChannel,
)
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch


class ObservabilityDecoder(nn.Module):
    """Dense per-modality observability head over the BEV grid."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Initialize the observability head.

        Args:
            lead_config: Root config tree.
        """
        super().__init__()
        self.lead_config = lead_config
        self.config = lead_config.policy.transfuser

        self.net = nn.Sequential(
            nn.Conv2d(
                self.config.bev_feature_channels,
                self.config.observability_head_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.observability_head_channels,
                NUM_OBSERVABILITY_CHANNELS,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    self.config.lidar_height_pixel // self.config.bev_downsample_factor,
                    self.config.lidar_width_pixel // self.config.bev_downsample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

    def forward(
        self,
        bev_feature_grid: jt.Float[torch.Tensor, "B C H W"],
    ) -> jt.Float[torch.Tensor, "B M cell_h cell_w"]:
        """Predict per-modality observability logits over the BEV cell grid.

        Args:
            bev_feature_grid: Top-down BEV feature grid from the encoder.

        Returns:
            One logit per cell per modality; apply a sigmoid for a probability.
        """
        return self.net(bev_feature_grid)

    def compute_loss(
        self,
        pred: jt.Float[torch.Tensor, "B M cell_h cell_w"],
        data: TransfuserForwardBatch,
        loss: TaskLosses,
        log: AuxiliaryLog,
    ) -> None:
        """Accumulate the masked observability loss.

        Args:
            pred: Predicted observability logits.
            data: Batch holding the targets and their mask.
            loss: Dict the computed loss is stored in.
            log: Dict the metrics are stored in.
        """
        if not self.config.use_observability:
            return

        target = data["observability"].to(pred.device, non_blocking=True)
        mask = data["observability_mask"].to(pred.device, non_blocking=True)

        with autocast(device_type="cuda", enabled=False):
            elementwise = F.binary_cross_entropy_with_logits(
                pred.float(),
                target.float(),
                reduction="none",
            )
            # Cells no actor was measured in carry no target; normalizing by the
            # supervised count keeps the loss scale independent of how crowded
            # the scene happens to be.
            supervised = mask.sum()
            observability_loss = (elementwise * mask).sum() / supervised.clamp(min=1.0)

        loss["loss_observability"] = observability_loss

        gradient_step = data.get("current_gradient_step")
        log_every = self.lead_config.training.experiment.log_scalars_every_n_steps
        if gradient_step is not None and ((gradient_step + 1) % log_every) == 0:
            log["outputs/observability_supervised_cells"] = supervised.item()
            probability = torch.sigmoid(pred.float())
            for channel in ObservabilityChannel:
                channel_mask = mask[:, channel]
                count = channel_mask.sum().clamp(min=1.0)
                name = channel.name.lower()
                log[f"metric/observability_{name}_error"] = (
                    (
                        (probability[:, channel] - target[:, channel]).abs()
                        * channel_mask
                    )
                    .sum()
                    .div(count)
                    .item()
                )
                log[f"outputs/observability_{name}_target"] = (
                    (target[:, channel] * channel_mask).sum().div(count).item()
                )
