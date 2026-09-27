"""Every catalogue fault on one real training frame, for looking at.

Tests check that a fault does what its code says; they cannot check that what
it says looks like the phenomenon it is named after. This renders the exact
model inputs of one frame -- the stitched three-camera image and the BEV LiDAR
raster, read through the training dataset -- under each fault at two
severities, so the catalogue can be judged by eye before anything is scored
against it.

Usage (from the worktree, PYTHONPATH at its src):
    fault_gallery.py OUT_DIR [frame_index]
"""
import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from lead.config import LeadConfig  # noqa: E402
from lead.policy.transfuser.transfuser import Transfuser  # noqa: E402
from lead.policy.transfuser.utils.fault_catalog import (  # noqa: E402
    CAMERA_FAULTS,
    LIDAR_FAULTS,
    FaultGeometry,
    apply_fault,
)

SEL = pathlib.Path("/home/new_drive/razaghi/lead/thesis-artifacts/provenance/new_subset/selected_frames_town.txt")
SEVERITIES = (0.5, 1.0)


def sample(index: int) -> dict:
    names = sorted(line.split("/")[1] for line in SEL.read_text().split() if line.strip())
    config = LeadConfig()
    config.training.data.read_from_cache_store = True
    config.training.data.py123d_log_names = names[:40]
    dataset = Transfuser(config).build_dataset()
    item = dataset[index % len(dataset)]
    # A TransfuserTrainingSample: the model inputs are attributes, not keys.
    return {"rgb": torch.as_tensor(item.rgb)[None].float(),
            "rasterized_lidar": torch.as_tensor(item.rasterized_lidar)[None].float()}


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    clean = sample(index)
    geometry = FaultGeometry()

    rows = len(CAMERA_FAULTS) + 1
    fig, axes = plt.subplots(rows, len(SEVERITIES), figsize=(16, 2.1 * rows))
    for col, severity in enumerate(SEVERITIES):
        axes[0, col].imshow(clean["rgb"][0].permute(1, 2, 0).clamp(0, 255).byte().numpy())
        axes[0, col].set_title("clean" if col == 0 else "", loc="left", fontsize=9)
        for row, fault in enumerate(CAMERA_FAULTS, start=1):
            batch = {k: v.clone() for k, v in clean.items()}
            img = apply_fault(batch, fault, torch.tensor([severity]), torch.Generator().manual_seed(0),
                              17, geometry)["rgb"][0]
            axes[row, col].imshow(img.permute(1, 2, 0).clamp(0, 255).byte().numpy())
            axes[row, col].set_title(f"{fault}  severity {severity}", loc="left", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "camera_faults.png", dpi=110)

    rows = len(LIDAR_FAULTS) + 1
    fig, axes = plt.subplots(rows, len(SEVERITIES), figsize=(8, 3.3 * rows))
    for col, severity in enumerate(SEVERITIES):
        axes[0, col].imshow(clean["rasterized_lidar"][0, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[0, col].set_title("clean" if col == 0 else "", loc="left", fontsize=9)
        for row, fault in enumerate(LIDAR_FAULTS, start=1):
            batch = {k: v.clone() for k, v in clean.items()}
            raster = apply_fault(batch, fault, torch.tensor([severity]), torch.Generator().manual_seed(0),
                                 17, geometry)["rasterized_lidar"][0, 0]
            axes[row, col].imshow(raster.numpy(), cmap="magma", vmin=0, vmax=1)
            axes[row, col].set_title(f"{fault}  severity {severity}", loc="left", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "lidar_faults.png", dpi=110)
    print(f"rgb {tuple(clean['rgb'].shape)}  lidar {tuple(clean['rasterized_lidar'].shape)}")
    print(f"wrote {out}/camera_faults.png and {out}/lidar_faults.png")


if __name__ == "__main__":
    main()
