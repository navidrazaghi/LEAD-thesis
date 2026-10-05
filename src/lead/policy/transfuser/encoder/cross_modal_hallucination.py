"""Predict the LiDAR branch's BEV grid from the camera branch, and stand in for
it when the LiDAR is damaged.

The thesis measures that shifting attention away from a damaged modality does
not improve driving. One reading of that is mechanical: reweighting redistributes
what the encoders produced, and when a sensor is destroyed there is nothing left
in its share to redistribute. Under that reading the lever is not routing but
representation -- which is also what the ladder shows, since the rung that
changes what the encoders learn (the degradation curriculum) is the one that
worked while the rung that changes routing did not.

This module tests that reading directly. Instead of moving weight off the
damaged LiDAR grid, it rebuilds the grid from the camera and substitutes it.

Two things bound what this can do, and both are geometry rather than method:

The input cameras face forward, so a third of the BEV -- everything behind the
ego -- projects into no camera at all. ``bev_cells_in_image`` reports which
cells are visible and only those are predicted, supervised or substituted. On
the shipped configuration that is 80 of 120 token cells; the rest keep whatever
the damaged LiDAR gave, because inventing them would be fabrication.

And a cell is projected at one reference height, so the correspondence is exact
only for content at that height. The lift is a fixed geometric sampling, not a
learned view transform: the calibration is known and there is no reason to make
the network rediscover it.
"""

import jaxtyping as jt
import torch
from torch import nn
from torch.nn import functional as F

from lead.config import LeadConfig
from lead.policy.transfuser.encoder import fusion_geometry


class CrossModalHallucination(nn.Module):
    """Rebuilds the LiDAR BEV grid from camera features, where a camera sees it."""

    def __init__(
        self,
        lead_config: LeadConfig,
        image_channels: int,
        lidar_channels: int,
    ) -> None:
        """Build the refinement head.

        Args:
            lead_config: Root config tree, for the calibration.
            image_channels: Channels of the camera feature map it reads.
            lidar_channels: Channels of the LiDAR feature map it predicts.
        """
        super().__init__()
        self.lead_config = lead_config
        # Keyed by grid shape: it depends only on that and the fixed rig, and
        # recomputing it per forward would put a NumPy round trip in the step.
        # A plain dict rather than lru_cache on the method, which would hold a
        # reference to the module in a cache that outlives it.
        self._correspondence_cache: dict[
            tuple[int, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        hidden = max(image_channels // 2, lidar_channels)
        # Pointwise, then one 3x3, then pointwise. The sampling already placed
        # every cell where it belongs, so the head only has to translate camera
        # appearance into whatever the LiDAR encoder represents there; a wide
        # receptive field would let it smear content across cells instead.
        self.refine = nn.Sequential(
            nn.Conv2d(image_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, lidar_channels, kernel_size=1),
        )

    def _correspondence(
        self,
        rows: int,
        cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where each BEV cell of an ``(rows, cols)`` grid reads in the image.

        Args:
            rows: BEV grid rows to build the correspondence for.
            cols: BEV grid columns.

        Returns:
            The sampling grid in ``grid_sample`` coordinates, shaped
            ``(1, rows, cols, 2)``, and the visibility mask, shaped
            ``(1, 1, rows, cols)``.
        """
        cached = self._correspondence_cache.get((rows, cols))
        if cached is not None:
            return cached

        config = self.lead_config.policy.transfuser
        normalized, visible = fusion_geometry.bev_cells_in_image(
            self.lead_config,
            config.deformable_reference_height_meter,
            rows=rows,
            cols=cols,
        )
        # grid_sample wants [-1, 1] with x first; bev_cells_in_image gives [0, 1].
        grid = torch.from_numpy(normalized).float().mul(2.0).sub(1.0)
        mask = torch.from_numpy(visible).float()
        built = (
            grid.reshape(1, rows, cols, 2),
            mask.reshape(1, 1, rows, cols),
        )
        self._correspondence_cache[(rows, cols)] = built
        return built

    def correspondence(
        self,
        rows: int,
        cols: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The cached correspondence, moved to where the features live.

        Args:
            rows: BEV grid rows.
            cols: BEV grid columns.
            device: Device of the features being sampled.
            dtype: Dtype of the features being sampled.

        Returns:
            The sampling grid and the visibility mask.
        """
        grid, mask = self._correspondence(rows, cols)
        return grid.to(device=device, dtype=dtype), mask.to(device=device, dtype=dtype)

    def forward(
        self,
        image_features: jt.Float[torch.Tensor, "B C H W"],
        rows: int,
        cols: int,
    ) -> tuple[
        jt.Float[torch.Tensor, "B L R S"],
        jt.Float[torch.Tensor, "1 1 R S"],
    ]:
        """Predict the LiDAR BEV grid the camera can account for.

        Args:
            image_features: The camera branch's feature map.
            rows: Rows of the LiDAR grid to predict.
            cols: Columns of the LiDAR grid to predict.

        Returns:
            The prediction, zeroed on cells no camera sees, and the visibility
            mask that zeroed them.
        """
        grid, mask = self.correspondence(
            rows,
            cols,
            image_features.device,
            image_features.dtype,
        )
        sampled = F.grid_sample(
            image_features,
            grid.expand(image_features.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return self.refine(sampled) * mask, mask


def hallucination_loss(
    predicted: jt.Float[torch.Tensor, "B L R S"],
    target: jt.Float[torch.Tensor, "B L R S"],
    mask: jt.Float[torch.Tensor, "1 1 R S"],
) -> torch.Tensor:
    """Squared error between the prediction and the LiDAR branch, where visible.

    The target is detached: this trains the camera to describe what the LiDAR
    branch represents, and must not pull the LiDAR branch toward something
    easier to guess.

    Args:
        predicted: The camera's prediction of the LiDAR grid.
        target: The LiDAR branch's own features at the same point.
        mask: Which cells a camera sees; the rest carry no supervision.

    Returns:
        Mean squared error over the supervised cells.
    """
    error = (predicted - target.detach()).pow(2) * mask
    # Divide by the supervised cells rather than all of them, so the loss does
    # not shrink just because a third of the grid is unseen.
    return error.sum() / (mask.sum() * predicted.shape[0] * predicted.shape[1]).clamp(min=1.0)


def blend(
    lidar_features: jt.Float[torch.Tensor, "B L R S"],
    predicted: jt.Float[torch.Tensor, "B L R S"],
    mask: jt.Float[torch.Tensor, "1 1 R S"],
    reliability: float,
) -> jt.Float[torch.Tensor, "B L R S"]:
    """Substitute the prediction where the LiDAR is unreliable and visible.

    Args:
        lidar_features: What the LiDAR branch produced, damage and all.
        predicted: The camera's stand-in.
        mask: Which cells a camera sees. Outside it the LiDAR is kept whatever
            its state, because nothing else is available.
        reliability: How much of the LiDAR to keep, in ``[0, 1]``.

    Returns:
        The blended grid.
    """
    weight = mask * (1.0 - reliability)
    return lidar_features * (1.0 - weight) + predicted * weight


def lidar_reliability(lead_config: LeadConfig) -> float:
    """How much of the LiDAR grid the harness left intact, in ``[0, 1]``.

    Read from the degradation the harness applied rather than predicted, so a
    hallucination run measures the substitution and not an estimator on top of
    it. The two questions are separable and this module answers only the first.

    Args:
        lead_config: Root config tree.

    Returns:
        1 when the LiDAR is untouched, 0 when it is destroyed.

    Raises:
        ValueError: If a spatial degrade_family is active, whose severity varies
            over the grid and has no scalar to stand for it.
    """
    inference = lead_config.evaluation.inference
    if inference.degrade_family != "none":
        raise ValueError(
            f"hallucinate_missing_lidar has no scalar reliability for the "
            f"spatial family '{inference.degrade_family}'.",
        )
    if inference.degrade_modality != "lidar":
        return 1.0
    return float(max(0.0, 1.0 - inference.degrade_severity))
