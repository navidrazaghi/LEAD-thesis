"""Turn the observability estimate into speed, instead of into attention weights.

The observability head was shown to be informative: its output tracks sensor
degradation, and the gate fed from the same supervision moved attention mass by
a factor of eleven to thirty-eight in the right direction. What it did not do
was drive better. The measured reason is that the gate acts on attention logits,
and attention's authority over the token that leaves a fusion block is only
about four tenths, so a large shift in what the model reads became a small shift
in what it does -- and under full camera destruction the small shift was
negative.

This actuates the same signal where the authority is not fractional. If the
model cannot resolve the road ahead, the car slows down. Nothing about the
network changes: the checkpoint is frozen, the forward pass is untouched, and
the only thing this reads is a head the model already computes.

Two choices in the signal are worth stating because they are not the obvious
ones. The modalities are combined by taking the better of the two per cell
rather than averaging them, because redundancy is the point of carrying two
sensors -- a cell one modality resolves is resolved, and averaging would call it
half-seen. And only the corridor the ego is about to drive through counts,
because a mean over the whole grid is dominated by cells behind the car and out
to the sides, where being unable to see costs nothing.
"""

import functools
import typing

import jaxtyping as jt
import numpy as np
import torch

from lead.config import LeadConfig
from lead.policy.transfuser.decoder import waypoint_ensemble
from lead.policy.transfuser.encoder import fusion_geometry


def corridor_mask(
    lead_config: LeadConfig,
    cell_height: int,
    cell_width: int,
    device: torch.device,
) -> jt.Bool[torch.Tensor, "cell_h cell_w"]:
    """Which cells of the observability grid lie in the ego's driving corridor.

    Args:
        lead_config: Root config tree, for the BEV grid's extent in meters.
        cell_height: Rows of the observability grid.
        cell_width: Columns of the observability grid.
        device: Device to build the mask on.

    Returns:
        The mask, true where a cell is ahead of the ego and inside the corridor
        half-width.
    """
    config = lead_config.policy.transfuser
    governor = lead_config.evaluation.inference

    # Columns run along x, back to front, and rows along y, left to right. That
    # is the raster's own convention, stated where the observability targets are
    # rasterized, and getting it backwards is not something a constant test
    # field can reveal -- the mask stays non-empty and every symmetric check
    # still passes while the corridor points across the road.
    column_edges = torch.linspace(
        config.bev_min_x_meter, config.bev_max_x_meter, cell_width + 1, device=device,
    )
    row_edges = torch.linspace(
        config.bev_min_y_meter, config.bev_max_y_meter, cell_height + 1, device=device,
    )
    forward = 0.5 * (column_edges[:-1] + column_edges[1:])
    lateral = 0.5 * (row_edges[:-1] + row_edges[1:])

    ahead = (forward >= 0.0) & (forward <= governor.caution_corridor_length_meter)
    within = lateral.abs() <= governor.caution_corridor_half_width_meter
    return within.view(-1, 1) & ahead.view(1, -1)


def observability_caution(
    observability_logits: jt.Float[torch.Tensor, "bs n_modalities cell_h cell_w"],
    lead_config: LeadConfig,
) -> float:
    """How poorly the sensors resolve the road ahead, in ``[0, 1]``.

    Args:
        observability_logits: The head's per-modality logits over the BEV cell
            grid, as the frozen model computed them.
        lead_config: Root config tree.

    Returns:
        Zero when the corridor is resolved, one when it is not.

    Raises:
        ValueError: If the configured modality rule is not one of the three.
    """
    resolved = torch.sigmoid(observability_logits.float())
    rule = lead_config.evaluation.inference.caution_modality_rule
    if rule == "best":
        best, _ = resolved.max(dim=1)
    elif rule == "mean":
        best = resolved.mean(dim=1)
    elif rule == "worst":
        best, _ = resolved.min(dim=1)
    else:
        raise ValueError(
            f"caution_modality_rule must be 'best', 'mean' or 'worst', got {rule!r}.",
        )

    mask = corridor_mask(
        lead_config,
        best.shape[-2],
        best.shape[-1],
        best.device,
    )
    if not bool(mask.any()):
        # A corridor that lands outside the grid would otherwise divide by zero
        # and report perfect confidence, which is the wrong way to fail.
        return 1.0
    corridor = best[..., mask]
    return float(1.0 - corridor.mean())


def target_speed_multiplier(
    caution: float,
    conservativeness: float,
    lead_config: LeadConfig,
) -> float:
    """Map caution and the calibrated conservativeness onto a speed factor.

    The calibrated scalar sets the slope rather than a cut-off: at zero the
    governor is inert and the policy drives exactly as the frozen checkpoint
    does, and as it rises the same caution buys progressively more slowing. That
    keeps the un-governed model inside the family, so the ablation compares a
    mechanism against its own absence rather than against a different tuning.

    Args:
        caution: How poorly the road ahead is resolved, in ``[0, 1]``.
        conservativeness: The calibrator's scalar; zero disables the governor.
        lead_config: Root config tree, for the floor.

    Returns:
        A multiplier in ``[floor, 1]`` to apply to the predicted target speed.
    """
    floor = lead_config.evaluation.inference.caution_speed_floor
    demand = min(max(caution * conservativeness, 0.0), 1.0)
    return float(1.0 - (1.0 - floor) * demand)


class CellCameraGeometry(typing.NamedTuple):
    """Where each BEV token sits in the stitched image, and how far away it is.

    Attributes:
        image_xy: Normalized ``(x, y)`` position in the stitched image, in
            ``[0, 1]``, of the camera that sees the cell most centrally.
        axial_depth: Distance along that camera's optical axis, in meters,
            which is the quantity a CARLA depth map records.
        visible: Whether any input camera sees the cell at all.
    """

    image_xy: jt.Float[torch.Tensor, "n 2"]
    axial_depth: jt.Float[torch.Tensor, " n"]
    visible: jt.Bool[torch.Tensor, " n"]


@functools.lru_cache(maxsize=4)
def _cell_camera_geometry_cached(
    config_id: int,
    lead_config: LeadConfig,
) -> CellCameraGeometry:
    """Build the projection table once per config; see ``cell_camera_geometry``."""
    del config_id
    specs = fusion_geometry.stitched_camera_specs(lead_config)
    centres = fusion_geometry.bev_cell_centres(lead_config)
    height = lead_config.policy.transfuser.deformable_reference_height_meter
    points = np.concatenate(
        [centres, np.full((centres.shape[0], 1), height)],
        axis=1,
    )

    count = centres.shape[0]
    best_xy = np.zeros((count, 2))
    best_depth = np.zeros(count)
    found = np.zeros(count, dtype=bool)
    best_offset = np.full(count, np.inf)

    for camera_index, spec in enumerate(specs):
        pixels, inside = fusion_geometry.project_to_camera(spec, points)
        # The depth a CARLA camera records is along its optical axis, not the
        # straight-line distance, so the comparison has to use the same one.
        rotation = fusion_geometry.world_to_camera_rotation(spec["rot"])
        in_camera = (points - np.asarray(spec["pos"])) @ rotation.T
        axial = in_camera[:, 0]

        offset = np.abs(pixels[:, 0] - spec["width"] / 2.0)
        wins = inside & (offset < best_offset)
        best_xy[wins, 0] = pixels[wins, 0] + camera_index * spec["width"]
        best_xy[wins, 1] = pixels[wins, 1]
        best_depth[wins] = axial[wins]
        best_offset[wins] = offset[wins]
        found |= wins

    transfuser = lead_config.policy.transfuser
    best_xy[:, 0] /= transfuser.final_image_width
    best_xy[:, 1] /= transfuser.final_image_height
    return CellCameraGeometry(
        torch.as_tensor(best_xy, dtype=torch.float32),
        torch.as_tensor(best_depth, dtype=torch.float32),
        torch.as_tensor(found),
    )


def cell_camera_geometry(lead_config: LeadConfig) -> CellCameraGeometry:
    """The image position and axial depth of every BEV token.

    Fixed by the rig and the token grids, so it is built once and reused; the
    loader re-expresses a perturbated rig into the nominal one, so the network
    always observes this same calibration.

    Args:
        lead_config: Root config tree.

    Returns:
        The projection table.
    """
    return _cell_camera_geometry_cached(id(lead_config), lead_config)


def cross_modal_caution(
    depth_prediction: jt.Float[torch.Tensor, "bs img_h img_w"],
    rasterized_lidar: jt.Float[torch.Tensor, "bs 1 bev_h bev_w"],
    lead_config: LeadConfig,
) -> float:
    """How much the camera's depth and the LiDAR's returns contradict each other.

    This is the signal the observability head cannot supply. That head reports
    per modality, so with one sensor destroyed the other still resolves the
    scene and the combined estimate correctly says nothing is wrong. Useful, and
    also blind exactly where a single failure has to be caught. Two sensors
    looking at one world give a second signal that needs no label at all: they
    have to agree about what is there, and a broken one stops agreeing.

    Two contradictions are counted, and they catch opposite failures. A cell the
    LiDAR reports occupied while the camera predicts free space beyond it means
    the camera is seeing through something -- what a dimmed or blurred image
    does. A cell the camera puts a surface at while the LiDAR returns nothing
    means the sweep has lost it -- what dropout does.

    Args:
        depth_prediction: The depth head's metric output, in meters.
        rasterized_lidar: The BEV density raster the model was given.
        lead_config: Root config tree.

    Returns:
        The fraction of examined corridor cells whose two modalities disagree;
        zero when there is nothing to examine.
    """
    transfuser = lead_config.policy.transfuser
    governor = lead_config.evaluation.inference
    rows = transfuser.lidar_bev_grid_rows
    cols = transfuser.lidar_bev_grid_cols
    device = depth_prediction.device

    geometry = cell_camera_geometry(lead_config)
    image_xy = geometry.image_xy.to(device)
    axial_depth = geometry.axial_depth.to(device)
    visible = geometry.visible.to(device)

    examined = visible & corridor_mask(lead_config, rows, cols, device).reshape(-1)
    # Beyond the depth head's far plane its output carries no information, so a
    # disagreement there would be an artefact of the quantization rather than of
    # the sensors.
    examined = examined & (axial_depth < transfuser_depth_far_plane(lead_config))
    if not bool(examined.any()):
        return 0.0

    # Sample the predicted depth where each cell lands, in the [-1, 1] frame
    # grid_sample uses.
    grid = (2.0 * image_xy - 1.0).view(1, -1, 1, 2).expand(
        depth_prediction.shape[0], -1, -1, -1,
    )
    sampled = torch.nn.functional.grid_sample(
        depth_prediction.unsqueeze(1).float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(depth_prediction.shape[0], -1)

    occupancy = torch.nn.functional.adaptive_avg_pool2d(
        rasterized_lidar.float(),
        (rows, cols),
    ).reshape(rasterized_lidar.shape[0], -1)

    tolerance = governor.caution_depth_tolerance_meter
    lidar_occupied = occupancy > 0.0
    sees_through = sampled > axial_depth + tolerance
    sees_a_surface = (sampled - axial_depth).abs() <= tolerance

    disagree = (lidar_occupied & sees_through) | (~lidar_occupied & sees_a_surface)
    counted = disagree[:, examined]
    return float(counted.float().mean())


def ensemble_caution(
    member_waypoints: jt.Float[torch.Tensor, "bs members waypoints 2"],
    lead_config: LeadConfig,
) -> float:
    """How much the ensemble's readouts disagree about where to go, in ``[0, 1]``.

    The spread is in meters and the governor wants a fraction, so it is divided
    by the spread at which the plan is considered fully unresolved. That scale
    is the one number here that cannot be derived: it says how much
    disagreement is a lot, in the units of the thing being disagreed about. It
    is a config field for that reason, and the calibrator's scalar is what
    decides how much slowing any given fraction buys -- so getting the scale
    somewhat wrong changes the signal's gain, not its direction.

    The ensemble producing these must have been in evaluation mode. Its decoder
    layers carry dropout, and in training mode the spread includes a different
    mask per member rather than a different opinion -- a real uncertainty
    measure, but not the one this claims to be reporting.

    Args:
        member_waypoints: Every member's predicted waypoints.
        lead_config: Root config tree.

    Returns:
        Zero when the members agree exactly, one when they are at least the
        configured spread apart.
    """
    scale = lead_config.evaluation.inference.caution_spread_meter
    spread = float(waypoint_ensemble.ensemble_spread(member_waypoints).mean())
    return float(min(max(spread / scale, 0.0), 1.0))


def transfuser_depth_far_plane(lead_config: LeadConfig) -> float:
    """The far plane the depth labels were quantized against, in meters.

    Args:
        lead_config: Root config tree.

    Returns:
        The far plane.
    """
    return float(lead_config.expert.storage.save_depth_max_meters)


def surrogate_risk_event(
    caution: float,
    ego_speed_mps: float,
    lead_config: LeadConfig,
) -> bool:
    """Whether this tick counts as a risk event for the calibrator.

    An infraction is the thing worth avoiding but it is far too rare and far too
    late to calibrate on: a run produces a handful, and each one arrives after
    the decisions that caused it. This stands in for it with the condition those
    decisions share -- carrying speed into a stretch of road the model cannot
    resolve. It is a surrogate and is named one; what it has to be is frequent,
    observable at the tick, and monotone in the thing it stands for.

    Args:
        caution: How poorly the road ahead is resolved, in ``[0, 1]``.
        ego_speed_mps: Current speed.
        lead_config: Root config tree.

    Returns:
        True when the tick is one the calibrator should count against.
    """
    governor = lead_config.evaluation.inference
    unresolved = caution >= governor.caution_risk_threshold
    moving = ego_speed_mps >= governor.caution_risk_speed_mps
    return bool(unresolved and moving)
