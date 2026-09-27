"""Hash what every pre-existing degradation path produces, for a fixed input and seed.

Run once against the main checkout and once against the worktree carrying the
fault catalogue. Identical hashes mean the old corruptions are untouched, not
just believed to be.
"""
import hashlib

import torch

import lead
from lead.policy.transfuser.utils.sensor_degradation import (
    apply_sensor_degradation,
    degrade_batch,
    degrade_batch_family,
)


def batch(uint8: bool) -> dict:
    g = torch.Generator().manual_seed(123)
    rgb = torch.rand(4, 3, 96, 288, generator=g) * 255.0
    return {
        "rgb": rgb.round().to(torch.uint8) if uint8 else rgb,
        "rasterized_lidar": (torch.rand(4, 1, 320, 384, generator=g) < 0.25).float() * 0.6,
        "ego_speed": torch.full((4, 1), 6.0),
        "target_point": torch.rand(4, 2, generator=g) * 10.0,
    }


def digest(b: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(b):
        value = b[key]
        if isinstance(value, torch.Tensor):
            h.update(key.encode())
            h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


print("lead imported from", lead.__file__)
results = {}
for modality in ("camera", "lidar"):
    out = degrade_batch(batch(False), modality, 1.0, torch.Generator().manual_seed(7))
    results[f"eval {modality}:1.0"] = digest(out)
for family in ("occlusion", "ego_state"):
    out = degrade_batch_family(batch(False), family, 1.0, torch.Generator().manual_seed(7))
    results[f"eval family {family}:1.0"] = digest(out)
# The campaign's curriculum v2, draw for draw from the global stream.
for name, kwargs in (
    ("train curriculum v1 (p 0.5)", dict(probability=0.5, max_severity=1.0)),
    ("train curriculum v2 (campaign)", dict(probability=0.30, max_severity=1.0, independent_modalities=True,
                                            full_failure_probability=0.25, misalignment_probability=0.10)),
):
    torch.manual_seed(99)
    try:
        out = apply_sensor_degradation(batch(True), **kwargs)
        results[name] = digest(out)
    except Exception as error:  # a batch missing a key this path needs
        results[name] = f"skipped ({type(error).__name__}: {error})"[:90]
for key, value in results.items():
    print(f"{key:34s} {value}")
