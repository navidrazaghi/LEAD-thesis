"""Pull each rung's training loss out of its offline W&B log.

The runs were logged offline and never synced, so the curves exist only inside
``outputs/<rung>/wandb/offline-run-*/run-*.wandb``. Nothing in the repository
read them, and the thesis has shown the ladder's results without ever showing
that the models were converging while they produced them.

What is written is the sum of the scaled losses -- the quantity the optimiser
actually descends -- averaged per epoch, plus the semantic head on its own,
which is the one the convergence check in ``docs/baseline_convergence.md`` uses.
Per-epoch rather than per-step because the step series is noisy enough to hide
the trend at figure size, and because the epoch is what the pretrain and the
posttrain are counted in.

Two details the file format forces:

The metric name lives in ``item.nested_key``, not ``item.key``, which is empty
on every record here. Reading ``key`` alone returns one nameless series.

There is no total-loss key. The scaled heads are the terms of the objective, so
they are summed; the unscaled ones are the same quantities before their weights
and would not be what was minimised.

A rung trained in two stages has a second directory with the ``_post`` suffix,
and its epochs continue the first's rather than restarting, so the two are
concatenated with the stage recorded.
"""

import csv
import glob
import json
import pathlib
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "training_curves.csv"

_SCALED = "losses/scaled_"
# The unscaled head, because that is the series the thesis's divergence
# subsection quotes. The objective stays scaled: the scaled terms are what the
# optimiser actually sums.
_SEMANTIC = "losses/unscaled_semantic"


def read_history(path):
    """Every history record in one offline run, as dictionaries.

    Args:
        path: The ``run-*.wandb`` file.

    Yields:
        One dictionary per logged step.
    """
    store = datastore.DataStore()
    store.open_for_scan(str(path))
    while True:
        raw = store.scan_data()
        if raw is None:
            return
        record = pb.Record()
        record.ParseFromString(raw)
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


def per_epoch(path):
    """Mean objective and mean semantic loss for each epoch of one run.

    Args:
        path: The ``run-*.wandb`` file.

    Returns:
        ``{epoch: (objective, semantic, count)}``.
    """
    totals = {}
    for row in read_history(path):
        epoch = row.get("epoch")
        if epoch is None:
            continue
        terms = [v for k, v in row.items()
                 if k.startswith(_SCALED) and isinstance(v, (int, float))]
        if not terms:
            continue
        semantic = row.get(_SEMANTIC)
        objective, seen, total = totals.get(int(epoch), (0.0, 0.0, 0))
        totals[int(epoch)] = (
            objective + sum(terms),
            seen + (semantic if isinstance(semantic, (int, float)) else 0.0),
            total + 1,
        )
    return {
        epoch: (objective / count, semantic / count, count)
        for epoch, (objective, semantic, count) in totals.items()
    }


def main():
    rows = []
    for directory in sorted((ROOT / "outputs").iterdir()):
        if not directory.is_dir():
            continue
        logs = sorted(glob.glob(str(directory / "wandb" / "offline-run-*" / "run-*.wandb")))
        if not logs:
            continue
        name = directory.name
        stage = "post" if name.endswith("_post") else "pre"
        rung = name[:-5] if stage == "post" else name
        for log in logs:
            curve = per_epoch(pathlib.Path(log))
            for epoch in sorted(curve):
                objective, semantic, count = curve[epoch]
                rows.append({
                    "run": name,
                    "rung": rung,
                    "stage": stage,
                    "epoch": epoch,
                    "objective": round(objective, 6),
                    "semantic": round(semantic, 6),
                    "steps": count,
                })
        print(f"  {name:<36} {len(curve):2d} epochs")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run", "rung", "stage", "epoch", "objective", "semantic", "steps"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
