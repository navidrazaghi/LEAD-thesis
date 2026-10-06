"""The head predicting the weather visibility class the expert conditions on.

The dense observability head answers a geometric question: how much of a place
did each sensor resolve, given occlusion and range. This one answers an
environmental question the geometry cannot reach -- how far anything can be
seen at all, because of fog, rain or darkness -- and the two are different
enough that a model can be right about one and blind to the other.

It exists because of an asymmetry that is already in the repository. The
expert reads ``WeatherVisibility`` and drives differently under it: under
LIMITED it shortens the lane-change transition, under VERY_LIMITED it shortens
it further and lowers its target speed. The student imitates those decisions
without ever seeing the variable they were conditioned on, so it has to
attribute them to whatever else happened to be in the frame. Predicting the
class makes the variable available to the encoder instead of leaving it as an
unexplained cause of expert behaviour.

The label costs nothing new. Every log already records it per frame, and
``dataset._VERBATIM_DRIVING_META_KEYS`` already carries it into the batch; no
run so far has read it. In the 450-log training subset it is close to
balanced across its four classes, so it is usable as a classification target
without reweighting.

The head is off by default: no result in this project was trained with it.
"""

import jaxtyping as jt
import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F

from lead.api.abstract_policy import AuxiliaryLog, TaskLosses
from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch

# CLEAR, OK, LIMITED, VERY_LIMITED. Zero is the clearest, which is the
# opposite of the direction observability runs in, and worth stating here
# because the two are easy to confuse when reading the loss.
NUM_VISIBILITY_CLASSES = 4


class VisibilityDecoder(nn.Module):
    """Frame-level weather-visibility classifier over the BEV feature grid."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Initialize the visibility head.

        Args:
            lead_config: Root config tree.
        """
        super().__init__()
        self.lead_config = lead_config
        self.config = lead_config.policy.transfuser

        # Visibility is a property of the frame, not of a place in it, so the
        # grid is pooled away rather than decoded. That also keeps the head
        # small enough that any effect it has cannot be attributed to capacity.
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.config.bev_feature_channels,
                      self.config.weather_visibility_head_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.weather_visibility_head_channels,
                      NUM_VISIBILITY_CLASSES),
        )

    def forward(
        self,
        bev_feature_grid: jt.Float[torch.Tensor, "B C H W"],
    ) -> jt.Float[torch.Tensor, "B K"]:
        """Predict one logit per visibility class.

        Args:
            bev_feature_grid: Top-down BEV feature grid from the encoder.

        Returns:
            Unnormalized scores over the four classes.
        """
        return self.net(bev_feature_grid)

    def compute_loss(
        self,
        pred: jt.Float[torch.Tensor, "B K"],
        data: TransfuserForwardBatch,
        loss: TaskLosses,
        log: AuxiliaryLog,
    ) -> None:
        """Accumulate the visibility classification loss.

        Args:
            pred: Predicted class scores.
            data: Batch holding the recorded visibility class.
            loss: Dict the computed loss is stored in.
            log: Dict the metrics are stored in.
        """
        if not self.config.use_weather_visibility:
            return

        # The key travels through the batch verbatim, so it arrives as
        # whatever the collate made of a Python int. as_tensor accepts all of
        # those without committing this head to one of them.
        target = torch.as_tensor(
            data["visual_visibility"],
            device=pred.device,
        ).reshape(-1).long()

        with autocast(device_type="cuda", enabled=False):
            loss["loss_weather_visibility"] = F.cross_entropy(
                pred.float(),
                target,
            )
        log["weather_visibility_accuracy"] = (
            (pred.argmax(dim=-1) == target).float().mean().detach()
        )
