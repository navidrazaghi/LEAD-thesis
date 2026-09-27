"""Training loss of the seed-2 consistency run, against seeds 0 and 1.

The objective is computed exactly as scripts/common/training_curves.py computes
it for every other run in the repository -- the per-epoch mean of the summed
scaled loss terms, the quantity the optimiser descends -- so the seed-2 line is
on the same scale as the seed-0 and seed-1 curves in
thesis-artifacts/results/training_curves_consistency_seeds.csv.

The run is still training, so its log is copied before it is read (the writer
appends as it goes, and the last record may be half written), and the epoch in
progress is drawn as an open marker: it is a partial mean, not a finished one.

Usage: seed2_loss_plot.py OUT.png [OUT.csv]
"""
import csv
import json
import pathlib
import shutil
import sys
import tempfile
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal import datastore  # noqa: E402

R = pathlib.Path("/home/new_drive/razaghi/lead")
RUNS = {
    "pretrain": R / "outputs/rung2a_diverse_curriculum2_seed2",
    "posttrain": R / "outputs/rung2a_diverse_consistency_seed2_post31",
}
REFERENCE = R / "thesis-artifacts/results/training_curves_consistency_seeds.csv"
SCALED = "losses/scaled_"

# Categorical slots 1-3 of the validated reference palette (light mode). Slot 3
# sits below 3:1 on the surface, so every line carries a direct label.
COLOR = {"seed0": "#2a78d6", "seed1": "#eb6834", "seed2": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def history(path):
    """History rows of an offline W&B run, tolerating a half-written tail."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / path.name
        shutil.copyfile(path, copy)
        store = datastore.DataStore()
        store.open_for_scan(str(copy))
        while True:
            try:
                raw = store.scan_data()
            except Exception:  # the record being written when the copy was taken
                return
            if raw is None:
                return
            record = pb.Record()
            try:
                record.ParseFromString(raw)
            except Exception:
                return
            if record.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in record.history.item:
                name = "/".join(item.nested_key) if item.nested_key else item.key
                try:
                    row[name] = json.loads(item.value_json)
                except (ValueError, TypeError):
                    continue
            yield row


def seed2_series(directory):
    """Per-epoch mean objective, and the per-step objective, for one stage."""
    logs = sorted(directory.glob("wandb/offline-run-*/run-*.wandb"))
    if not logs:
        return {}, []
    per_epoch = defaultdict(list)
    steps = []
    for row in history(logs[-1]):
        epoch = row.get("epoch")
        terms = [v for k, v in row.items() if k.startswith(SCALED) and isinstance(v, (int, float))]
        if epoch is None or not terms:
            continue
        per_epoch[int(epoch)].append(sum(terms))
        steps.append((int(epoch), sum(terms)))
    return {e: sum(v) / len(v) for e, v in per_epoch.items()}, steps


def reference():
    curves = defaultdict(dict)
    for r in csv.DictReader(open(REFERENCE)):
        curves[(r["seed"], r["stage"])][int(r["epoch"])] = float(r["train_objective"])
    return curves


def smooth(values, window=25):
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def epoch_panel(ax, curves, stage, seed2, title, finished=False):
    style(ax, title)
    ends = {}
    for seed in ("seed0", "seed1"):
        c = curves.get((seed, stage), {})
        if not c:
            continue
        xs = sorted(c)
        ax.plot(xs, [c[x] for x in xs], color=COLOR[seed], linewidth=2, marker="o", markersize=4,
                label=seed.replace("seed", "seed "))
        ends[seed] = (xs[-1], c[xs[-1]])
    # A stage that has written its last checkpoint is complete; only a stage
    # still running has an epoch whose mean is partial.
    done = ({e: v for e, v in seed2.items() if finished or e < max(seed2)} if seed2 else {})
    if done:
        xs = sorted(done)
        ax.plot(xs, [done[x] for x in xs], color=COLOR["seed2"], linewidth=2.6, marker="o",
                markersize=5, label="seed 2 (this run)")
        ends["seed2"] = (xs[-1], done[xs[-1]])
    # End labels, stacked by height: the runs finish within a hair of each other,
    # so each label gets its own row instead of its own point's height.
    offsets = {1: [0], 2: [7, -9], 3: [11, 0, -11]}[len(ends)] if ends else []
    for (seed, (x, y)), dy in zip(sorted(ends.items(), key=lambda kv: -kv[1][1]), offsets):
        mine = seed == "seed2"
        ax.annotate(seed.replace("seed", "seed "), (x, y), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=9, color=INK if mine else INK2,
                    fontweight="bold" if mine else "normal")
    if seed2 and not finished:
        last = max(seed2)
        ax.plot([last], [seed2[last]], marker="o", markersize=7, markerfacecolor=SURFACE,
                markeredgecolor=COLOR["seed2"], markeredgewidth=2, linestyle="none",
                label="seed 2, epoch in progress")
    if not seed2:
        ax.text(0.5, 0.92, "seed 2: not started yet", transform=ax.transAxes, ha="center",
                fontsize=9, color=INK2, style="italic")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training objective (sum of scaled losses)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")


def main():
    out = pathlib.Path(sys.argv[1])
    curves = reference()
    pre_epoch, pre_steps = seed2_series(RUNS["pretrain"])
    post_epoch, post_steps = seed2_series(RUNS["posttrain"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    pre_done = (RUNS["pretrain"] / "model_0030.pth").exists()
    post_done = (RUNS["posttrain"] / "model_0030.pth").exists()
    epoch_panel(axes[0], curves, "pretrain", pre_epoch, "Pretrain: loss per epoch", pre_done)

    ax = axes[1]
    live = post_steps or pre_steps
    stage = "post-train" if post_steps else "pretrain"
    style(ax, f"Seed 2 right now ({stage}): every logged step")
    if live:
        ys = [v for _, v in live]
        ax.plot(range(len(ys)), ys, color=COLOR["seed2"], linewidth=0.8, alpha=0.35)
        ax.plot(range(len(ys)), smooth(ys), color=COLOR["seed2"], linewidth=2.4,
                label="rolling mean, 25 logs")
        # Epoch boundaries, so the steps read against the left panel.
        previous = live[0][0]
        for i, (epoch, _) in enumerate(live):
            if epoch != previous:
                ax.axvline(i, color=GRID, linewidth=1)
                ax.text(i, max(ys), f" ep {epoch}", fontsize=7.5, color=INK2, va="top")
                previous = epoch
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    ax.set_xlabel("logged step (one per 50 optimiser steps)")
    ax.set_ylabel("training objective")

    epoch_panel(axes[2], curves, "posttrain", post_epoch, "Post-train: loss per epoch", post_done)

    fig.suptitle("Dense + curriculum v2 + consistency 0.1: training loss, seed 2 against seeds 0 and 1",
                 x=0.01, ha="left", fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)

    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["seed", "stage", "epoch", "train_objective", "complete"])
            for stage_name, series in (("pretrain", pre_epoch), ("posttrain", post_epoch)):
                for e in sorted(series):
                    writer.writerow(["seed2", stage_name, e, round(series[e], 6), e < max(series)])

    for stage_name, series, finished in (("pretrain", pre_epoch, pre_done), ("posttrain", post_epoch, post_done)):
        for e in sorted(series):
            ref = [curves.get((s, stage_name), {}).get(e) for s in ("seed0", "seed1")]
            ref_txt = "  ".join(f"{x:.4f}" if x is not None else "   -  " for x in ref)
            flag = "" if finished or e < max(series) else "  (in progress)"
            print(f"{stage_name:9s} epoch {e:2d}  seed2 {series[e]:.4f}   seed0/seed1 {ref_txt}{flag}")


if __name__ == "__main__":
    main()
