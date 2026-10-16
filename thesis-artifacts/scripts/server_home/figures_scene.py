# -*- coding: utf-8 -*-
"""One recorded frame: what the three input cameras and the LiDAR saw at once.

Unlike every other figure in this thesis, this one is drawn from data rather
than from a schema, so it cannot be regenerated without the dataset. It was
produced on the machine that held the 450-log training subset, and the log and
frame are named below so the same picture can be redrawn from the same source.

    scenario  CrossJunctionDefectTrafficLight
    log       Town04_Rep0_route_002171_route0_08_01_21_45_33
    frame     44 of 47
    timestamp 11,050,000 us, which the LiDAR sweep matches exactly

Two things about the stored format matter for reading the plot. The 123D IMU
frame negates the CARLA lateral axis, so positive y is to the *left* here and
the axis is labelled to say so. And the ego origin is the IMU, not the camera
or the LiDAR, so the near-field disc is measured from that origin.

Run with the dataset present:

    python figures_scene.py /path/to/data/lead/123D/logs/normal_view
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402
from py123d.api.scene.arrow.modalities.arrow_lidar import _decode_lidar_binary  # noqa: E402
from py123d.common.io.camera.jpeg_camera_io import decode_image_from_jpeg_binary  # noqa: E402

LOG = "Town04_Rep0_route_002171_route0_08_01_21_45_33"
FRAME = 44
CAMERAS = ("pcam_l0", "pcam_f0", "pcam_r0")
TITLES = ("front-left  −57.5°", "front  0°", "front-right  +57.5°")

CAMERA = "#c1663a"
LIDAR = "#2f6f9f"
NEUTRAL = "#5b6b7a"
LINE = "#8a99a8"


def _column(path: pathlib.Path, name: str):
    """Read one Arrow column whole."""
    with pa.memory_map(str(path)) as source:
        return ipc.open_file(source).read_all().column(name)


def sensor_scene(root: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Draw the frame named at the top of this module.

    Args:
        root: The ``logs/normal_view`` directory of the dataset.
        out: Directory to write the figure into.

    Returns:
        The written path.
    """
    matches = list(root.glob(f"*/{LOG}"))
    if not matches:
        sys.exit(f"log {LOG} not found under {root}")
    log = matches[0]

    images = [
        decode_image_from_jpeg_binary(
            _column(log / f"camera.{camera}.arrow", f"camera.{camera}.data")[FRAME].as_py())
        for camera in CAMERAS
    ]
    stamp = _column(log / "camera.pcam_f0.arrow", "camera.pcam_f0.timestamp_us")[FRAME].as_py()
    stamps = np.array(_column(log / "lidar.lidar_top.arrow",
                              "lidar.lidar_top.timestamp_us").to_pylist())
    row = int(np.argmin(np.abs(stamps - stamp)))
    points, _ = _decode_lidar_binary(
        _column(log / "lidar.lidar_top.arrow", "lidar.lidar_top.data")[row].as_py())
    # Anything more than half a metre above the lowest returns is treated as
    # structure rather than road surface; it only sets the two point colours.
    above = points[:, 2] > np.percentile(points[:, 2], 5) + 0.5

    figure = plt.figure(figsize=(9.6, 7.8))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 2.4], hspace=0.10, wspace=0.03)
    for index, (image, title) in enumerate(zip(images, TITLES)):
        axes = figure.add_subplot(grid[0, index])
        axes.imshow(image)
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_edgecolor(CAMERA)
            spine.set_linewidth(1.0)
        axes.set_title(title, fontsize=9, color=CAMERA, pad=4)

    axes = figure.add_subplot(grid[1, :])
    inside = ((points[:, 0] > -32) & (points[:, 0] < 64)
              & (points[:, 1] > -40) & (points[:, 1] < 40))
    axes.add_patch(patches.Rectangle((-40, 0), 80, 64, facecolor=CAMERA,
                                     alpha=0.07, zorder=0))
    axes.add_patch(patches.Rectangle((-40, -32), 80, 32, facecolor="#8a99a8",
                                     alpha=0.11, zorder=0))
    for value in range(-40, 41, 8):
        axes.plot([value, value], [-32, 64], color=LINE, lw=0.35, alpha=0.5, zorder=1)
    for value in range(-32, 65, 8):
        axes.plot([-40, 40], [value, value], color=LINE, lw=0.35, alpha=0.5, zorder=1)
    ground, structure = inside & ~above, inside & above
    axes.scatter(points[ground, 1], points[ground, 0], s=0.25, c="#c9d3dc",
                 linewidths=0, zorder=2)
    axes.scatter(points[structure, 1], points[structure, 0], s=0.45, c=LIDAR,
                 linewidths=0, zorder=2)
    # The camera arc ends at 87.5 degrees, which on this scale is almost the
    # lateral axis; drawing it shows why the split sits where it does.
    slope = np.tan(np.radians(87.5))
    for sign in (1, -1):
        axes.plot([0, sign * 40], [0, 40 / slope], color=CAMERA, lw=1.0,
                  ls=(0, (5, 3)), zorder=4)
    axes.plot([-40, 40], [0, 0], color=NEUTRAL, lw=1.3, zorder=4)
    axes.add_patch(patches.Rectangle((-1.0, -1.2), 2.0, 4.6, facecolor="white",
                                     edgecolor=NEUTRAL, lw=1.0, zorder=6))
    axes.set_xlim(40, -40)
    axes.set_ylim(-32, 64)
    axes.set_aspect("equal")
    axes.set_xlabel("lateral, metres — positive is left in this frame",
                    fontsize=9, color=NEUTRAL)
    axes.set_ylabel("forward, metres", fontsize=9, color=NEUTRAL)
    axes.tick_params(labelsize=8, colors=NEUTRAL)
    for spine in axes.spines.values():
        spine.set_visible(False)

    label = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.5)
    axes.text(0, 59, "80 cells: lidar and camera", ha="center", fontsize=9.5,
              color=CAMERA, bbox=label, zorder=7)
    axes.text(0, -29, "40 cells: lidar alone, no camera reaches here", ha="center",
              fontsize=9.5, color=NEUTRAL, bbox=label, zorder=7)
    axes.text(-26, 5.5, "edge of the 175° camera arc", ha="center", fontsize=8,
              color=CAMERA, bbox=label, zorder=7)
    axes.annotate("an empty disc around the vehicle:\nno returns from the near ground",
                  xy=(2.8, -2.6), xytext=(21, -12), fontsize=8, color=LIDAR,
                  ha="center", bbox=label, zorder=7,
                  arrowprops=dict(arrowstyle="->", color=LIDAR, lw=0.9))
    axes.text(-3.4, 2.4, "ego", ha="left", fontsize=8, color=NEUTRAL, zorder=7)

    path = out / "fig_5_8_sensor_scene.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  wrote {path.name}")
    return path


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "data/lead/123D/logs/normal_view")
    sensor_scene(root, pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "."))
