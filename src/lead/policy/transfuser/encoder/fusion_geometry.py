"""Calibration-derived correspondence between the image and BEV token grids.

The deformable fusion operator needs a reference point for every query on every
modality. On its own modality that is the query's own cell; on the other one
there is no free answer, because a perspective image grid and a top-down BEV
grid share no coordinate system. This module supplies the answer the rig
already knows: where each BEV cell lands in the stitched camera image, and
where each image token's ray meets the ground.

Two facts make a single precomputed table correct for every sample. The token
grids are fixed, because the backbone pools both branches onto a fixed anchor
grid before fusion. And the rig is fixed from the model's side: when data
collection perturbates the cameras, the loader re-expresses labels and LiDAR in
the perturbated rig's own frame (see
:mod:`lead.log_reader.view_geometry`), so the network always observes the
nominal calibration.

The projection math takes plain camera specs rather than the config tree, so it
can be exercised on its own; only :func:`calibrated_reference_points` reads the
config.
"""

import math

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch

from lead.api.py123d_log_api import CAMERA_ID_BY_LEAD_INDEX
from lead.common.geometry import euler_degrees_to_rotation_matrix
from lead.config import LeadConfig
from lead.config.expert.sensor_rig_config import CameraSpec

# A point exactly on the camera plane has no projection; anything closer to the
# camera than this counts as behind it.
_MIN_DEPTH_METER = 1e-3


def world_to_camera_rotation(
    rotation_degrees: list[float],
) -> jt.Float[npt.NDArray, "3 3"]:
    """Rotation taking ego-frame directions into the camera's own axes.

    :func:`~lead.common.geometry.euler_degrees_to_rotation_matrix` builds the
    camera-to-ego rotation from a mounting pose, so the ego-to-camera direction
    is its transpose.

    Args:
        rotation_degrees: The camera's ``(roll, pitch, yaw)`` mounting rotation.

    Returns:
        The 3x3 ego-to-camera rotation matrix.
    """
    roll, pitch, yaw = rotation_degrees
    return euler_degrees_to_rotation_matrix(roll, pitch, yaw).T


def _focal_lengths(spec: CameraSpec) -> tuple[float, float]:
    """Pinhole focal lengths in pixels, from the spec's vertical field of view.

    Args:
        spec: The camera's calibration.

    Returns:
        The ``(focal_x, focal_y)`` focal lengths in pixels.
    """
    focal_y = spec["height"] / (2.0 * math.tan(math.radians(spec["fov"]) / 2.0))
    return focal_y * (spec["width"] / spec["height"]), focal_y


def project_to_camera(
    spec: CameraSpec,
    points_ego: jt.Float[npt.NDArray, "n 3"],
) -> tuple[jt.Float[npt.NDArray, "n 2"], jt.Bool[npt.NDArray, " n"]]:
    """Project ego-frame points into one camera's pixel coordinates.

    The ego frame is CARLA's: x forward, y right, z up. The camera's own axes
    are x right, y down, z forward.

    Args:
        spec: The camera's calibration.
        points_ego: Points in the ego frame.

    Returns:
        The pixel coordinates, and whether each point is in front of the camera
        and inside its image bounds.
    """
    rotation = world_to_camera_rotation(spec["rot"])
    translated = np.asarray(points_ego, dtype=np.float64) - np.asarray(spec["pos"])
    in_camera = translated @ rotation.T

    right = in_camera[:, 1]
    down = -in_camera[:, 2]
    forward = in_camera[:, 0]

    focal_x, focal_y = _focal_lengths(spec)
    in_front = forward > _MIN_DEPTH_METER
    depth = np.where(in_front, forward, 1.0)
    u = focal_x * right / depth + spec["width"] / 2.0
    v = focal_y * down / depth + spec["height"] / 2.0

    inside = (
        in_front & (u >= 0.0) & (u < spec["width"]) & (v >= 0.0) & (v < spec["height"])
    )
    return np.stack([u, v], axis=1), inside


def ground_points_of_pixels(
    spec: CameraSpec,
    pixels: jt.Float[npt.NDArray, "n 2"],
    ground_height_meter: float,
) -> tuple[jt.Float[npt.NDArray, "n 2"], jt.Bool[npt.NDArray, " n"]]:
    """Intersect each pixel's viewing ray with a horizontal plane.

    Args:
        spec: The camera's calibration.
        pixels: Pixel coordinates in this camera's own image.
        ground_height_meter: Height of the plane in the ego frame.

    Returns:
        The ego-frame ``(x, y)`` intersections, and whether each ray meets the
        plane in front of the camera. Rays at or above the horizon do not.
    """
    focal_x, focal_y = _focal_lengths(spec)
    pixels = np.asarray(pixels, dtype=np.float64)
    right = (pixels[:, 0] - spec["width"] / 2.0) / focal_x
    down = (pixels[:, 1] - spec["height"] / 2.0) / focal_y

    # Back to ego axes: forward is the camera's z, right its x, up its -y.
    directions_camera = np.stack([np.ones_like(right), right, -down], axis=1)
    directions = directions_camera @ world_to_camera_rotation(spec["rot"])

    position = np.asarray(spec["pos"], dtype=np.float64)
    vertical = directions[:, 2]
    descends = vertical < -_MIN_DEPTH_METER
    distance = np.where(
        descends,
        (ground_height_meter - position[2]) / np.where(descends, vertical, -1.0),
        0.0,
    )
    hits = position[None, :] + distance[:, None] * directions
    return hits[:, :2], descends & (distance > 0.0)


def stitched_camera_specs(lead_config: LeadConfig) -> list[CameraSpec]:
    """The calibrations of the model's input cameras, in stitch order.

    Args:
        lead_config: Root config tree.

    Returns:
        One spec per input camera, left to right as the model sees them.

    Raises:
        ValueError: If an input camera has no calibration in the rig.
    """
    rig_index_by_camera_id = {
        camera_id: index for index, camera_id in CAMERA_ID_BY_LEAD_INDEX.items()
    }
    cameras = lead_config.expert.sensor_rig.cameras
    specs = []
    for camera_id in lead_config.policy.transfuser.input_cameras:
        index = rig_index_by_camera_id.get(camera_id)
        if index is None or index > len(cameras):
            raise ValueError(f"Camera {camera_id} has no calibration in the rig.")
        specs.append(cameras[index - 1])
    return specs


def bev_cell_centres(lead_config: LeadConfig) -> jt.Float[npt.NDArray, "n 2"]:
    """Ego-frame ``(x, y)`` centre of every BEV token, in token order.

    The BEV raster runs its columns along x, back to front, and its rows along
    y, left to right; the tokens flatten row-major, as
    :meth:`~lead.policy.transfuser.encoder.transfuser_backbone.GPT.forward`
    reads them.

    Args:
        lead_config: Root config tree.

    Returns:
        The cell centres in metres.
    """
    config = lead_config.policy.transfuser
    rows = config.lidar_bev_grid_rows
    cols = config.lidar_bev_grid_cols
    x_span = config.bev_max_x_meter - config.bev_min_x_meter
    y_span = config.bev_max_y_meter - config.bev_min_y_meter

    x = config.bev_min_x_meter + (np.arange(cols) + 0.5) * (x_span / cols)
    y = config.bev_min_y_meter + (np.arange(rows) + 0.5) * (y_span / rows)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)


def image_token_pixels(
    lead_config: LeadConfig,
) -> jt.Float[npt.NDArray, "n 2"]:
    """Stitched-image pixel centre of every image token, in token order.

    Args:
        lead_config: Root config tree.

    Returns:
        The pixel centres in the stitched image.
    """
    config = lead_config.policy.transfuser
    rows = config.img_vert_anchors
    cols = config.img_horz_anchors
    u = (np.arange(cols) + 0.5) * (config.final_image_width / cols)
    v = (np.arange(rows) + 0.5) * (config.final_image_height / rows)
    grid_v, grid_u = np.meshgrid(v, u, indexing="ij")
    return np.stack([grid_u.reshape(-1), grid_v.reshape(-1)], axis=1)


def bev_cells_in_image(
    lead_config: LeadConfig,
    reference_height_meter: float,
) -> tuple[jt.Float[npt.NDArray, "n 2"], jt.Bool[npt.NDArray, " n"]]:
    """Where each BEV token lands in the stitched image, normalized to [0, 1].

    A cell can fall inside more than one camera where the fields of view
    overlap; the most central view wins, being the least distorted and the
    least likely to be clipped.

    Args:
        lead_config: Root config tree.
        reference_height_meter: Height above the ego's ground plane at which
            the cell is projected.

    Returns:
        The normalized ``(x, y)`` image positions, and whether each cell is
        visible in any input camera at all.
    """
    config = lead_config.policy.transfuser
    specs = stitched_camera_specs(lead_config)
    centres = bev_cell_centres(lead_config)
    points = np.concatenate(
        [centres, np.full((centres.shape[0], 1), reference_height_meter)],
        axis=1,
    )

    best = np.zeros((centres.shape[0], 2))
    found = np.zeros(centres.shape[0], dtype=bool)
    # Distance from the camera's principal point, lower being more central.
    best_offset = np.full(centres.shape[0], np.inf)

    for camera_index, spec in enumerate(specs):
        pixels, inside = project_to_camera(spec, points)
        offset = np.abs(pixels[:, 0] - spec["width"] / 2.0)
        wins = inside & (offset < best_offset)
        best[wins, 0] = pixels[wins, 0] + camera_index * spec["width"]
        best[wins, 1] = pixels[wins, 1]
        best_offset[wins] = offset[wins]
        found |= wins

    best[:, 0] /= config.final_image_width
    best[:, 1] /= config.final_image_height
    return best, found


def image_tokens_in_bev(
    lead_config: LeadConfig,
    reference_height_meter: float,
) -> tuple[jt.Float[npt.NDArray, "n 2"], jt.Bool[npt.NDArray, " n"]]:
    """Where each image token's ray meets the plane, normalized to [0, 1].

    Args:
        lead_config: Root config tree.
        reference_height_meter: Height of the plane the rays are cast onto.

    Returns:
        The normalized ``(x, y)`` BEV-grid positions, and whether each ray both
        meets the plane ahead of the camera and lands inside the BEV crop.
    """
    config = lead_config.policy.transfuser
    specs = stitched_camera_specs(lead_config)
    pixels = image_token_pixels(lead_config)

    ground = np.zeros((pixels.shape[0], 2))
    valid = np.zeros(pixels.shape[0], dtype=bool)
    for camera_index, spec in enumerate(specs):
        start = camera_index * spec["width"]
        owned = (pixels[:, 0] >= start) & (pixels[:, 0] < start + spec["width"])
        if not owned.any():
            continue
        own_pixels = pixels[owned] - np.array([start, 0.0])
        hits, descends = ground_points_of_pixels(
            spec,
            own_pixels,
            reference_height_meter,
        )
        ground[owned] = hits
        valid[owned] = descends

    # The grid axes: x runs along the columns, y along the rows.
    x_span = config.bev_max_x_meter - config.bev_min_x_meter
    y_span = config.bev_max_y_meter - config.bev_min_y_meter
    normalized = np.stack(
        [
            (ground[:, 0] - config.bev_min_x_meter) / x_span,
            (ground[:, 1] - config.bev_min_y_meter) / y_span,
        ],
        axis=1,
    )
    inside = (
        (normalized[:, 0] >= 0.0)
        & (normalized[:, 0] < 1.0)
        & (normalized[:, 1] >= 0.0)
        & (normalized[:, 1] < 1.0)
    )
    return normalized, valid & inside


def calibrated_reference_points(
    lead_config: LeadConfig,
    default_reference_points: jt.Float[torch.Tensor, "1 T L 2"],
    reference_height_meter: float,
) -> tuple[jt.Float[torch.Tensor, "1 T L 2"], jt.Bool[torch.Tensor, " T"]]:
    """Seed the cross-modal reference points from the rig's calibration.

    Own-modality references are left alone: a query's own cell is already the
    right anchor there. Cross-modality references are replaced wherever the
    projection is defined, and keep the passed-in default — the other grid's
    centre — wherever it is not, which is the case for BEV cells outside every
    camera's field of view and for image tokens looking at or above the horizon.

    Args:
        lead_config: Root config tree.
        default_reference_points: The operator's geometric defaults, whose
            token order and layout the result matches.
        reference_height_meter: Height above the ground plane at which the two
            grids are put into correspondence.

    Returns:
        The seeded reference points, and a per-token mask of which tokens got a
        calibrated cross-modal reference.
    """
    config = lead_config.policy.transfuser
    num_image_tokens = config.img_vert_anchors * config.img_horz_anchors

    reference = default_reference_points.clone()
    bev_in_image, bev_in_image_valid = bev_cells_in_image(
        lead_config,
        reference_height_meter,
    )
    image_in_bev, image_in_bev_valid = image_tokens_in_bev(
        lead_config,
        reference_height_meter,
    )

    # Image tokens are level 0 and BEV tokens level 1, matching the order the
    # fusion transformer concatenates them in. So an image token's cross-modal
    # reference is its ground hit on level 1, and a BEV cell's is its
    # projection on level 0.
    image_tokens = torch.from_numpy(np.flatnonzero(image_in_bev_valid))
    reference[0, image_tokens, 1] = torch.from_numpy(
        image_in_bev[image_in_bev_valid],
    ).to(reference.dtype)

    bev_tokens = torch.from_numpy(np.flatnonzero(bev_in_image_valid)) + num_image_tokens
    reference[0, bev_tokens, 0] = torch.from_numpy(
        bev_in_image[bev_in_image_valid],
    ).to(reference.dtype)

    valid = torch.from_numpy(
        np.concatenate([image_in_bev_valid, bev_in_image_valid]),
    )
    return reference, valid
