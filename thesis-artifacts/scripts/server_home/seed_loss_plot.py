# -*- coding: utf-8 -*-
"""Training-loss figure for the three seed replicates of dense + curriculum v2 + consistency.

Same reader as live_loss_plot.py (the per-epoch objective the optimiser descends).
Seed 0's pretrain is the dense + curriculum v2 run, which the consistency post-train
started from; seeds 1 and 2 pretrained their own. Read-only.

Usage: python seed_loss_plot.py [--loop SECONDS]
"""
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path.home()))
from live_loss_plot import curve  # noqa: E402

SEEDS = {
    "seed 0": ("rung2a_diverse_curriculum2", "rung2a_diverse_consistency_post31", "#2a78d6"),
    "seed 1": ("rung2a_diverse_curriculum2_seed1", "rung2a_diverse_consistency_seed1_post31", "#eb6834"),
    "seed 2": ("rung2a_diverse_curriculum2_seed2", "rung2a_diverse_consistency_seed2_post31", "#1baf7a"),
}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
OUT = pathlib.Path.home() / "loss_seeds.png"


def draw() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURF)
    status = []
    for label, (pre, post, color) in SEEDS.items():
        for ax, run in ((axes[0], pre), (axes[1], post)):
            c = curve(run)
            if not c:
                continue
            epochs = sorted(c)
            values = [c[e][0] for e in epochs]
            ax.plot(epochs, values, "-", lw=2, color=color, label=label)
            ax.annotate(
                f"{values[-1]:.3f}", (epochs[-1], values[-1]), xytext=(5, 0),
                textcoords="offset points", fontsize=9, color=color, va="center",
            )
            if run is post or len(epochs) < 31:
                status.append(f"{label} {'post-train' if run is post else 'pretrain'}: epoch {epochs[-1]}, {values[-1]:.4f}")
    for ax, title in zip(axes, ("Pretrain (31 epochs)", "Post-train with consistency (31 epochs)")):
        ax.set_facecolor(SURF)
        for x in (1, 3, 7, 15):
            ax.axvline(x, color="#dfe3e6", lw=1, zorder=0)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("training objective (sum of scaled losses)")
        ax.set_xlim(-0.5, 32)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
    note = " | ".join(status) or "no epoch logged yet"
    fig.suptitle(
        "Dense + curriculum v2 + consistency, three seeds — "
        f"updated {time.strftime('%m-%d %H:%M')}", fontsize=12, x=0.02, ha="left",
    )
    fig.text(0.02, 0.005, "Seed 0 post-trained from the dense + curriculum v2 pretrain; seeds 1 and 2 pretrained their own. Grey lines: LR warm restarts.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT, dpi=150, facecolor=SURF)
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
