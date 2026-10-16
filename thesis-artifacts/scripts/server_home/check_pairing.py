"""Pilot check: does the loader's own pairing code cover the new log, tick-aligned?"""
import pathlib, sys
from lead.log_reader import scene_index as si

ROOT = pathlib.Path.home() / "LEAD/lead/data/lead/123D"
name = sys.argv[1]
pairing = si.get_perturbated_view_pairing(ROOT, [name])
print("pairing logs:", None if pairing is None else len(pairing), "| covered:", pairing is not None and name in pairing.covered_logs())
logs_root = ROOT / "logs"
pert = si.build_log_scene_index(next((logs_root / "perturbated_view").glob(f"*/{name}")), pairing._scene_filter)
norm = si.build_log_scene_index(next((logs_root / "normal_view").glob(f"*/{name}")), pairing._scene_filter)
tp, tn = pert.anchor_timestamps_us, norm.anchor_timestamps_us
common = set(tp.tolist()) & set(tn.tolist())
print(f"anchors: normal {len(tn)}, perturbated {len(tp)}, common {len(common)}")
ts = sorted(common)[len(common) // 2]
scene = pairing.scene_at(name, int(ts))
print("scene_at OK:", type(scene).__name__)
cams = [m for m in dir(scene) if "camera" in m.lower() and not m.startswith("_")]
print("camera methods:", cams)
print("perturbated cameras:", scene.available_camera_names)
import numpy as np
for cid in scene.available_camera_ids:
    cam = scene.get_camera_at_iteration(0, cid)
    img = getattr(cam, "image", None)
    arr = np.asarray(img) if img is not None else None
    print(cid, type(cam).__name__, None if arr is None else (arr.shape, arr.dtype, round(float(arr.mean()), 1)))
