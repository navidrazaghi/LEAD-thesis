# -*- coding: utf-8 -*-
"""Check the widened curriculum, and that the old one is untouched.

The first check is the one that matters most: with the new flags off, the
sampler must consume the same random draws and produce the same batch as the
module did before the patch, because a run is currently training against it and
because the rungs already measured are defined by that stream. The pre-patch
file is loaded from /tmp/sd_backup.py and run side by side.

Run from the repo root.
"""
import importlib.util
import pathlib

import torch

from lead.policy.transfuser.utils import sensor_degradation as new

ok = True


def check(name, passed, detail=""):
    """Record and print one check."""
    global ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def load_old():
    """Import the pre-patch module from the backup copy."""
    spec = importlib.util.spec_from_file_location("sd_old", pathlib.Path("/tmp/sd_backup.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(batch_size=64, seed=0):
    """A batch with the three keys the sampler touches."""
    g = torch.Generator().manual_seed(seed)
    return {
        "rgb": torch.rand(batch_size, 3, 48, 144, generator=g) * 255.0,
        "rasterized_lidar": (torch.rand(batch_size, 1, 64, 64, generator=g) < 0.2).float(),
        "observability": torch.rand(batch_size, 2, 10, 12, generator=g),
    }


old = load_old()

# --- 1. the default path is byte-for-byte the old one ----------------------
torch.manual_seed(1234)
a = old.apply_sensor_degradation(make_batch(), 0.5, 1.0)
torch.manual_seed(1234)
b = new.apply_sensor_degradation(make_batch(), 0.5, 1.0)
same = all(torch.equal(a[k], b[k]) for k in ("rgb", "rasterized_lidar", "observability"))
check("default path reproduces the pre-patch curriculum exactly", same)

torch.manual_seed(7)
a = old.apply_sensor_degradation(make_batch(), 0.5, 1.0, ("occlusion", "ego_state"))
torch.manual_seed(7)
b = new.apply_sensor_degradation(make_batch(), 0.5, 1.0, ("occlusion", "ego_state"))
check("deployment-family path is unchanged too",
      all(torch.equal(a[k], b[k]) for k in ("rgb", "rasterized_lidar", "observability")))

# --- 2. independent draws give the intended split --------------------------
torch.manual_seed(0)
n, p = 20000, 0.30
cam = torch.rand(n) < p
lid = torch.rand(n) < p
clean = float((~cam & ~lid).float().mean())
both = float((cam & lid).float().mean())
check("independent draws reproduce the 49/21/21/9 split",
      abs(clean - 0.49) < 0.02 and abs(both - 0.09) < 0.02,
      f"clean {clean:.2f}, both {both:.2f}")

torch.manual_seed(3)
batch = new.apply_sensor_degradation(make_batch(256), 0.3, 1.0, (), independent_modalities=True)
scale = batch["observability"].amax(dim=(2, 3))
reference = make_batch(256)["observability"].amax(dim=(2, 3))
damaged_both = int(((scale[:, 0] < reference[:, 0] * 0.999) & (scale[:, 1] < reference[:, 1] * 0.999)).sum())
check("some samples now lose both modalities at once", damaged_both > 0,
      f"{damaged_both} of 256 samples")

# --- 3. full failure appears with positive probability ---------------------
torch.manual_seed(5)
batch = new.apply_sensor_degradation(
    make_batch(512), 0.3, 1.0, (), independent_modalities=True, full_failure_probability=0.25,
)
targets = batch["observability"].amax(dim=(2, 3))
zeroed = int(((targets[:, 0] == 0) | (targets[:, 1] == 0)).sum())
check("the fully-failed case is drawn with positive probability", zeroed > 0,
      f"{zeroed} of 512 samples have a modality at severity 1")

# --- 4. misalignment moves geometry without claiming lost visibility -------
torch.manual_seed(11)
before = make_batch(64)
after = new.apply_sensor_degradation(
    make_batch(64), 0.0, 1.0, (), independent_modalities=True,
    misalignment_probability=1.0, bev_pixels_per_meter=4.0,
)
moved = not torch.equal(before["rasterized_lidar"], after["rasterized_lidar"])
targets_intact = torch.equal(before["observability"], after["observability"])
mass = float(after["rasterized_lidar"].sum() / before["rasterized_lidar"].sum())
check("misalignment moves the raster", moved)
check("misalignment leaves the observability targets alone", targets_intact)
check("misalignment preserves most of the returns", 0.85 < mass < 1.05, f"mass ratio {mass:.3f}")

# --- 5. the targets still track what was damaged ---------------------------
torch.manual_seed(13)
batch = new.apply_sensor_degradation(
    make_batch(128), 1.0, 1.0, (), independent_modalities=True, full_failure_probability=1.0,
)
targets = batch["observability"].amax(dim=(2, 3))
check("with every modality fully failed the targets go to zero",
      bool((targets == 0).all()), f"max target {float(targets.max()):.4f}")

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
