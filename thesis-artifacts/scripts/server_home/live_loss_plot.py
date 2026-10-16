"""Live training-loss figure for the diverse baseline, beside the old baseline.

Reads the offline W&B logs with training_curves' own reader (same per-epoch
objective: the sum of the scaled loss terms the optimiser descends) and redraws
~/loss_live.png. Read-only: never touches the training run.

The old baseline trained without any perturbated view and on other data, so
its curve is a shape reference, not a level to beat.

Usage: python live_loss_plot.py [--loop SECONDS]
"""
import glob
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(ROOT / "scripts/common"))
from training_curves import per_epoch  # noqa: E402

RUNS = {
    "baseline, diverse data": ("rung0_diverse", "rung0_diverse_post31", "#c0392b"),
    "deformable, diverse data": ("rung2ad_diverse", "rung2ad_diverse_post31", "#2471a3"),
    "deformable + curriculum v2": ("rung2ad_diverse_curriculum2", "rung2ad_diverse_curriculum2_post31", "#1e8449"),
    "dense + curriculum v2": ("rung2a_diverse_curriculum2", "rung2a_diverse_curriculum2_post31", "#d68910"),
    "old baseline (450 logs)": ("rung0_lead_recipe", "rung0_lead_recipe_post31", "#aab7b8"),
}
OUT = pathlib.Path.home() / "loss_live.png"


def curve(run: str) -> dict:
    logs = sorted(glob.glob(str(ROOT / "outputs" / run / "wandb" / "offline-run-*" / "run-*.wandb")))
    merged = {}
    for log in logs:  # a resumed pretrain writes one log per launch
        merged.update(per_epoch(pathlib.Path(log)))
    return merged


def draw() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    status = []
    for label, (pre, post, color) in RUNS.items():
        for ax, run, stage in ((axes[0], pre, "pretrain"), (axes[1], post, "post-train")):
            c = curve(run)
            if not c:
                continue
            epochs = sorted(c)
            ax.plot(epochs, [c[e][0] for e in epochs], "o-", ms=3, color=color, label=label)
            if run.startswith("rung2ad_diverse_curriculum2"):
                status.append(f"curriculum2 {stage}: epoch {epochs[-1]}, loss {c[epochs[-1]][0]:.4f}")
    for ax, title in zip(axes, ("Pretrain (31 epochs)", "Post-train (31 epochs)")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("training objective (sum of scaled losses)")
        ax.set_xlim(-0.5, 30.5)
        ax.grid(alpha=0.3)
        ax.legend()
    note = " | ".join(status) or "curriculum v2 has not logged an epoch yet"
    fig.suptitle(f"Training loss, updated {time.strftime('%m-%d %H:%M')} -- {note}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    return note


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--loop":
        while True:
            try:
                print(time.strftime("%m-%d %H:%M"), draw(), flush=True)
            except Exception as error:  # a log mid-write must not kill the loop
                print(time.strftime("%m-%d %H:%M"), "skipped:", error, flush=True)
            time.sleep(int(sys.argv[2]))
    print(draw())
