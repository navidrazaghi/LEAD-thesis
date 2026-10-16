# -*- coding: utf-8 -*-
"""Export what the policy actually receives for one training sample.

  preview IDX...   save small camera thumbnails + stats, to pick a readable scene
  make IDX         save the presentation figures for one scene

Everything comes out of the same dataset and collate function the trainer uses,
with the rig-perturbation draw switched off so the camera is the nominal view.
The raw point cloud is re-read from the log for the same sample and accumulated
exactly as build_lidar_raster does before rasterizing.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/common"))
from analyze_gate import load_model  # noqa: E402
from lead.policy.transfuser.dataloader.point_cloud import accumulate_lidar_points  # noqa: E402
from lead.api.point_cloud_transforms import lidar_sweep_to_carla_ego_frame, radar_returns_to_carla_ego_frame  # noqa: E402
from lead.api.abstract_dataset import SceneLoadingSpec  # noqa: E402
from lead.policy.transfuser.utils.sensor_degradation import degrade_batch  # noqa: E402

OUT = pathlib.Path("/tmp/model_view")
OUT.mkdir(exist_ok=True)
mode, indices = sys.argv[1], [int(a) for a in sys.argv[2:]]

_, model = load_model(ROOT / "outputs/rung0_diverse_post31", torch.device("cuda:0"))
ds = model.build_dataset()
ds._scene_loader._perturbation_probability = 0.0
tc = model.lead_config.policy.transfuser


def batch_of(idx):
    return next(iter(DataLoader(Subset(ds, [idx]), batch_size=1, collate_fn=getattr(ds, "collate_fn", None))))


if mode == "preview":
    for idx in indices:
        b = batch_of(idx)
        rgb = b["rgb"][0].permute(1, 2, 0).numpy()
        occ = float((b["rasterized_lidar"][0, 0] > 0).float().mean())
        plt.imsave(OUT / f"thumb_{idx}.jpg", rgb[::3, ::3])
        print(idx, "town", b["town"][0], "mean brightness", round(float(rgb.mean()), 1),
              "lidar occupied", round(occ, 3), "speed", round(float(b["speed"][0]), 2),
              "weather", b["weather_setting"][0] if "weather_setting" in b else "?")
    sys.exit(0)

idx = indices[0]
b = batch_of(idx)
rgb = b["rgb"][0].permute(1, 2, 0).numpy()
raster = b["rasterized_lidar"][0, 0].numpy()

# raw points for the same tick, accumulated the way the raster was built
spec = SceneLoadingSpec.union(part.reads for part in ds._sample_parts.values())
scene = ds._scene_loader.read(idx, False, spec)
points = accumulate_lidar_points(
    {age: lidar_sweep_to_carla_ego_frame(l) for age, l in scene.lidar_sweeps.items()},
    None if scene.radar_sweeps is None else {age: radar_returns_to_carla_ego_frame(r)[:, :3].astype(np.float64) for age, r in scene.radar_sweeps.items()},
    scene.past_ego_positions, scene.past_ego_yaws, model.lead_config,
)

# 1. camera input: the stitched strip exactly as the network gets it
plt.imsave(OUT / "camera_input.png", rgb)

# 2. raw point cloud, top view, forward up, coloured by height
fig, ax = plt.subplots(figsize=(6, 7.2), dpi=160)
keep = (points[:, 2] >= tc.lidar_min_height_meter) & (points[:, 2] <= tc.lidar_max_height_meter)
p = points[keep]
sub = p[np.random.default_rng(0).choice(len(p), min(len(p), 120000), replace=False)]
sc = ax.scatter(sub[:, 1], sub[:, 0], c=sub[:, 2], s=0.25, cmap="viridis", vmin=-2.5, vmax=4.0, linewidths=0)
ax.add_patch(plt.Rectangle((tc.bev_min_y_meter, tc.bev_min_x_meter), tc.bev_max_y_meter - tc.bev_min_y_meter,
                           tc.bev_max_x_meter - tc.bev_min_x_meter, fill=False, ec="#F2A900", lw=1.5))
ax.plot(0, 0, marker="^", color="#D1495B", ms=9)
ax.set_xlim(-50, 50); ax.set_ylim(-45, 75); ax.set_aspect("equal")
ax.set_xlabel("lateral y (m)"); ax.set_ylabel("forward x (m)")
ax.set_facecolor("#1B1F24")
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02); cb.set_label("height z (m)")
fig.tight_layout(); fig.savefig(OUT / "lidar_raw_points.png", facecolor="white"); plt.close(fig)

# 2b. one raw sweep, unprocessed: ground kept, no radar, no accumulation; same frame as the raster
raw0 = lidar_sweep_to_carla_ego_frame(scene.lidar_sweeps[0])
fig, ax = plt.subplots(figsize=(5.2, 7.2), dpi=160)
ax.set_facecolor("#1B1F24")
inside = (raw0[:, 0] >= tc.bev_min_x_meter) & (raw0[:, 0] <= tc.bev_max_x_meter) & (raw0[:, 1] >= tc.bev_min_y_meter) & (raw0[:, 1] <= tc.bev_max_y_meter)
r = raw0[inside]
sc = ax.scatter(r[:, 1], r[:, 0], c=r[:, 2], s=0.35, cmap="viridis", vmin=-2.5, vmax=3.0, linewidths=0)
ax.plot(0, 0, marker="^", color="#D1495B", ms=9)
ax.set_xlim(tc.bev_min_y_meter, tc.bev_max_y_meter); ax.set_ylim(tc.bev_min_x_meter, tc.bev_max_x_meter); ax.set_aspect("equal")
ax.set_xlabel("lateral y (m)"); ax.set_ylabel("forward x (m)")
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label("height z (m)")
fig.tight_layout(); fig.savefig(OUT / "lidar_raw_sweep.png", facecolor="white"); plt.close(fig)
raw_sweep_points = int(len(raw0))

# 3. LiDAR input: the density raster, rows = y, cols = x; shown forward-up
img = np.flipud(raster.T)
extent = (tc.bev_min_y_meter, tc.bev_max_y_meter, tc.bev_min_x_meter, tc.bev_max_x_meter)
fig, ax = plt.subplots(figsize=(5.2, 7.2), dpi=160)
ax.imshow(img, cmap="inferno", extent=extent, vmin=0, vmax=1, interpolation="nearest")
tp = b["target_point"][0].numpy()
ax.plot(0, 0, marker="^", color="#2E9E6B", ms=9)
ax.plot(tp[1], tp[0], marker="*", color="#4FC3F7", ms=13)
ax.set_xlabel("lateral y (m)"); ax.set_ylabel("forward x (m)")
fig.tight_layout(); fig.savefig(OUT / "lidar_input_raster.png", facecolor="white"); plt.close(fig)

# 4. the same inputs after the evaluation's own destruction at severity 1
dev = torch.device("cuda:0")
g = lambda: torch.Generator(device=dev).manual_seed(0)
lid = degrade_batch({k: (v.clone().to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}, "lidar", 1.0, generator=g())
cam = degrade_batch({k: (v.clone().to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}, "camera", 1.0, generator=g())
cam_rgb = cam["rgb"][0].float().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
plt.imsave(OUT / "camera_destroyed.png", cam_rgb)
lid_img = np.flipud(lid["rasterized_lidar"][0, 0].float().cpu().numpy().T)
fig, ax = plt.subplots(figsize=(5.2, 7.2), dpi=160)
ax.imshow(lid_img, cmap="inferno", extent=extent, vmin=0, vmax=1, interpolation="nearest")
ax.plot(0, 0, marker="^", color="#2E9E6B", ms=9)
ax.set_xlabel("lateral y (m)"); ax.set_ylabel("forward x (m)")
fig.tight_layout(); fig.savefig(OUT / "lidar_destroyed_raster.png", facecolor="white"); plt.close(fig)

meta = {
    "index": idx, "town": b["town"][0], "log": ds.sample_log_name(idx),
    "rgb_shape": list(b["rgb"].shape[1:]), "raster_shape": list(b["rasterized_lidar"].shape[1:]),
    "raw_sweep_points": raw_sweep_points, "accumulated_points_total": int(len(points)), "raw_points_in_height_band": int(keep.sum()),
    "raster_occupied_fraction": float((raster > 0).mean()),
    "raster_occupied_fraction_lidar_destroyed": float((lid["rasterized_lidar"][0, 0] > 0).float().mean()),
    "speed_mps": float(b["speed"][0]), "target_point_xy": tp.tolist(),
    "radar_shape": list(b["radar1"].shape[1:]), "num_radars": sum(1 for k in b if k.startswith("radar") and k[5:].isdigit()),
    "bev_extent_m": {"x": [tc.bev_min_x_meter, tc.bev_max_x_meter], "y": [tc.bev_min_y_meter, tc.bev_max_y_meter]},
    "bev_pixels_per_meter": tc.bev_pixels_per_meter, "max_points_per_pixel": tc.max_lidar_points_per_bev_pixel,
    "lidar_sweeps_accumulated": len(scene.lidar_sweeps),
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
