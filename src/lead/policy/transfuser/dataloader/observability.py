"""Per-cell, per-modality observability targets, from what the expert measured.

While collecting data the expert counts, for every actor, how many LiDAR points
landed on it and how many of its pixels a camera could actually see, and reacts
to it only once both clear a per-class threshold. Those counts are recorded per
box per tick, so the visibility the expert reasoned with is already in the
dataset — as ``num_points`` on the box-detection stream and ``visible_pixels``
among the driving-meta box attributes.

This module rasterizes them into a target the student can be trained on: for
each cell of the BEV grid and each modality, how well that modality resolves
whatever occupies the cell. Supervision is sparse — only cells covered by a
measured actor carry a target — so the loss is masked, the same way the
CenterNet heatmap only supervises the cells it has boxes for.
"""

import enum

import numpy as np

from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserOutputs


class ObservabilityChannel(enum.IntEnum):
    """Channel order of the observability target and prediction."""

    CAMERA = 0
    LIDAR = 1


NUM_OBSERVABILITY_CHANNELS = len(ObservabilityChannel)

# Box classes whose visibility the expert measures. Traffic lights and stop
# signs are excluded: the expert reasons about them through the lanes they
# govern rather than through their own returns, so they carry no meaningful
# counts.
_MEASURED_CLASSES = frozenset({"car", "walker", "static"})

# A count of -1 marks a measurement the expert did not take for that box.
_MIN_VALID_COUNT = 0


def _thresholds(lead_config: LeadConfig, box_class: str) -> tuple[int, int]:
    """The expert's ``(visible pixels, LiDAR points)`` bar for a box class.

    Args:
        lead_config: Root config tree.
        box_class: The box's CARLA class name.

    Returns:
        The minimum counts at which the expert treats the actor as observed.
    """
    occlusion = lead_config.expert.occlusion
    if box_class == "walker":
        return (
            occlusion.pedestrian_min_num_visible_pixels,
            occlusion.pedestrian_min_num_lidar_points,
        )
    if box_class == "static":
        return (
            occlusion.static_prop_min_num_visible_pixels,
            occlusion.static_prop_min_num_lidar_points,
        )
    return (
        occlusion.vehicle_min_num_visible_pixels,
        occlusion.vehicle_min_num_lidar_points,
    )


def _cell_span(
    centre_meter: float,
    half_extent_meter: float,
    min_meter: float,
    meters_per_cell: float,
    num_cells: int,
) -> tuple[int, int]:
    """The half-open cell range a box covers along one axis, clipped to the grid.

    Args:
        centre_meter: Box centre along the axis, in the ego frame.
        half_extent_meter: Half the box's axis-aligned extent along the axis.
        min_meter: Lower bound of the BEV crop along the axis.
        meters_per_cell: Size of one cell along the axis.
        num_cells: Number of cells along the axis.

    Returns:
        The ``(start, stop)`` cell range, empty when the box misses the grid.
    """
    start = int(
        np.floor((centre_meter - half_extent_meter - min_meter) / meters_per_cell)
    )
    stop = int(
        np.ceil((centre_meter + half_extent_meter - min_meter) / meters_per_cell)
    )
    return max(start, 0), min(max(stop, start + 1), num_cells)


def build_observability_targets(
    view_boxes: list[dict],
    lead_config: LeadConfig,
) -> TransfuserOutputs:
    """Rasterize the expert's per-actor visibility counts onto the BEV grid.

    Where boxes overlap, the best-resolved one wins: a cell is as observable as
    the most visible thing in it.

    Args:
        view_boxes: The tick's view-frame box dicts, unfiltered.
        lead_config: Root config tree.

    Returns:
        The observability target and the mask of cells it supervises.
    """
    config = lead_config.policy.transfuser
    rows = config.lidar_height_pixel // config.bev_downsample_factor
    cols = config.lidar_width_pixel // config.bev_downsample_factor

    target = np.zeros([NUM_OBSERVABILITY_CHANNELS, rows, cols], dtype=np.float32)
    mask = np.zeros([NUM_OBSERVABILITY_CHANNELS, rows, cols], dtype=np.float32)

    # Columns run along x (back to front) and rows along y (left to right),
    # matching the BEV raster the network reads.
    meters_per_col = (config.bev_max_x_meter - config.bev_min_x_meter) / cols
    meters_per_row = (config.bev_max_y_meter - config.bev_min_y_meter) / rows

    for box in view_boxes:
        if box.get("class") not in _MEASURED_CLASSES:
            continue

        visible_pixels = box.get("visible_pixels", -1)
        lidar_points = box.get("num_points", -1)
        measured = np.array(
            [visible_pixels, lidar_points],
            dtype=np.float64,
        )
        # A negative count is an absent measurement, not an invisible actor.
        measurable = measured >= _MIN_VALID_COUNT
        if not measurable.any():
            continue

        pixel_bar, point_bar = _thresholds(lead_config, box["class"])
        bars = np.array([max(pixel_bar, 1), max(point_bar, 1)], dtype=np.float64)
        observability = np.clip(measured / bars, 0.0, 1.0)
        if not config.observability_soft_targets:
            observability = (observability >= 1.0).astype(np.float64)

        # The axis-aligned extent of the rotated footprint, which is what the
        # cell grid can represent.
        half_length, half_width = box["extent"][0], box["extent"][1]
        cos_yaw = abs(np.cos(box["yaw"]))
        sin_yaw = abs(np.sin(box["yaw"]))
        half_x = half_length * cos_yaw + half_width * sin_yaw
        half_y = half_length * sin_yaw + half_width * cos_yaw

        col_start, col_stop = _cell_span(
            box["position"][0],
            half_x,
            config.bev_min_x_meter,
            meters_per_col,
            cols,
        )
        row_start, row_stop = _cell_span(
            box["position"][1],
            half_y,
            config.bev_min_y_meter,
            meters_per_row,
            rows,
        )
        if col_start >= col_stop or row_start >= row_stop:
            continue

        for channel in ObservabilityChannel:
            if not measurable[channel]:
                continue
            patch = target[channel, row_start:row_stop, col_start:col_stop]
            np.maximum(patch, observability[channel], out=patch)
            mask[channel, row_start:row_stop, col_start:col_stop] = 1.0

    return {"observability": target, "observability_mask": mask}
