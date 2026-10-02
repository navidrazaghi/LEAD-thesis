"""Score the observability head against the expert labels it was trained on.

The thesis answers its second research question with "supporting but incomplete":
the head's output moves monotonically with degradation severity and separates by
modality, which an inert head would not do. That is a behavioural check, not a
measurement of accuracy, and the limitations say so. This is the measurement.

For every cell the expert actually measured -- the mask is sparse by
construction, only cells covered by a counted actor carry a label -- it compares
the head's probability against the label, per modality, and reports:

  Pearson and Spearman        does the head order cells the way the expert does
  MAE and RMSE                how far off it is in the label's own units
  calibration                 binned mean prediction against binned mean label,
                              and the gap averaged over bins
  a constant baseline         the same errors for predicting the label mean

The constant baseline is not decoration. A head that has learned nothing but the
average observability of a measured cell will still post a respectable MAE,
because the labels concentrate near one. Without something to beat, the error
numbers cannot be read at all.

WHAT THIS CANNOT DO, and why it is a required argument rather than a default:

Every log on disk went into training. There are 450 of them under
``logs/normal_view`` and the training config restricts nothing -- no log-name
list, no scene cap, no town filter -- so the model has seen every frame this
script can reach. Run without ``--log-names``, the numbers below are in-sample
and answer only "can the head fit the label at all", which is worth knowing
(a head that cannot fit in-sample has certainly learned nothing) but is not
evidence of generalisation. Held-out logs have to be fetched before the
generalisation question can be asked, and then named with ``--log-names``.

The script refuses to hide which of the two it ran.
"""

import argparse
import json
import math
import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import load_model, to_device  # noqa: E402

_CHANNELS = ("camera", "lidar")
_BINS = 10


def _rank(values: list[float]) -> list[float]:
    """Average ranks, so ties do not bias the rank correlation.

    Args:
        values: The sample.

    Returns:
        One rank per element, ties sharing their mean rank.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def _correlation(first: list[float], second: list[float]) -> float:
    """Pearson correlation, or nan when either side is constant.

    Args:
        first: One sample.
        second: The other, same length.

    Returns:
        The coefficient.
    """
    count = len(first)
    if count < 2:
        return float("nan")
    mean_a = sum(first) / count
    mean_b = sum(second) / count
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second, strict=True))
    spread_a = math.sqrt(sum((a - mean_a) ** 2 for a in first))
    spread_b = math.sqrt(sum((b - mean_b) ** 2 for b in second))
    if spread_a == 0 or spread_b == 0:
        return float("nan")
    return covariance / (spread_a * spread_b)


def _calibration(predicted: list[float], labels: list[float]) -> tuple:
    """Reliability of the head's probabilities, binned by prediction.

    Args:
        predicted: Head probabilities.
        labels: Expert labels in the same order.

    Returns:
        The per-bin table and the sample-weighted mean gap.
    """
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(_BINS)]
    for probability, label in zip(predicted, labels, strict=True):
        index = min(int(probability * _BINS), _BINS - 1)
        buckets[index].append((probability, label))
    table, weighted, total = [], 0.0, 0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_l = sum(v for _, v in bucket) / len(bucket)
        table.append((index / _BINS, (index + 1) / _BINS, len(bucket), mean_p, mean_l))
        weighted += abs(mean_p - mean_l) * len(bucket)
        total += len(bucket)
    return table, (weighted / total if total else float("nan"))


def collect(model, loader, batches: int, device) -> dict:
    """Gather every supervised cell's prediction and label.

    Args:
        model: The loaded policy.
        loader: Batches to read.
        batches: How many to read.
        device: Where to run.

    Returns:
        ``{channel: (predictions, labels)}``.
    """
    gathered = {name: ([], []) for name in _CHANNELS}
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            batch = to_device(batch, device)
            with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(batch)
            raw = getattr(predictions, "observability", None)
            if raw is None:
                raise SystemExit(
                    "this checkpoint has no observability head; run the probe on "
                    "a rung that carries one",
                )
            probability = torch.sigmoid(raw.float())
            target = batch["observability"].float()
            mask = batch["observability_mask"].float() > 0.5
            for channel, name in enumerate(_CHANNELS):
                keep = mask[:, channel]
                if not keep.any():
                    continue
                gathered[name][0].extend(
                    probability[:, channel][keep].cpu().tolist(),
                )
                gathered[name][1].extend(target[:, channel][keep].cpu().tolist())
    return gathered


def score(predicted: list[float], labels: list[float]) -> dict:
    """Every number this probe reports for one modality.

    Args:
        predicted: Head probabilities.
        labels: Expert labels.

    Returns:
        The scores, including the constant-predictor baseline.
    """
    count = len(labels)
    if count < 2:
        return {"cells": count}
    mean_label = sum(labels) / count
    errors = [p - v for p, v in zip(predicted, labels, strict=True)]
    flat = [mean_label - v for v in labels]
    table, gap = _calibration(predicted, labels)
    return {
        "cells": count,
        "label_mean": mean_label,
        "pred_mean": sum(predicted) / count,
        "pearson": _correlation(predicted, labels),
        "spearman": _correlation(_rank(predicted), _rank(labels)),
        "mae": sum(abs(e) for e in errors) / count,
        "rmse": math.sqrt(sum(e * e for e in errors) / count),
        "baseline_mae": sum(abs(e) for e in flat) / count,
        "baseline_rmse": math.sqrt(sum(e * e for e in flat) / count),
        "calibration_gap": gap,
        "calibration": table,
    }


def main() -> int:
    """Measure the observability head and print the result.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, metavar="DIR")
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--log-names",
        metavar="FILE",
        help="file of held-out log names, one per line; without it the run is "
             "in-sample and is labelled as such",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="read cached labels even when --log-names is given; only valid "
             "for logs the cache holds, and only useful for isolating the "
             "effect of caching from the effect of frame selection",
    )
    parser.add_argument("--json", metavar="FILE", help="also write the scores here")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    lead_config, model = load_model(pathlib.Path(arguments.model), device)

    held_out = None
    if arguments.log_names:
        names = [
            line.strip()
            for line in pathlib.Path(arguments.log_names).read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
        if not names:
            raise SystemExit("the log-name file is empty")
        try:
            lead_config.training.data.py123d_log_names = names
        except Exception as error:  # noqa: BLE001
            raise SystemExit(
                f"could not restrict the logs on this config ({error}); the "
                "held-out run cannot be guaranteed, so it is refused rather "
                "than run in-sample under a held-out label",
            ) from error
        # Held-out logs are not in the training cache and never will be, so
        # the cached path would either miss them or serve the wrong sample.
        # Computing live is slower and is the only correct option there.
        # --keep-cache exists for one job: restricting to logs that ARE cached,
        # so that caching can be varied while the frames stay fixed. Without it
        # a held-out run differs from an in-sample run in two ways at once.
        try:
            lead_config.training.data.read_from_cache_store = arguments.keep_cache
        except Exception as error:  # noqa: BLE001
            raise SystemExit(
                f"could not disable the cache store ({error}); held-out logs "
                "are not cached, so a cached read would not be a held-out read",
            ) from error
        held_out = names

    dataset = model.build_dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=getattr(dataset, "collate_fn", None),
        num_workers=arguments.workers,
    )
    gathered = collect(model, loader, arguments.batches, device)
    results = {name: score(*gathered[name]) for name in _CHANNELS}

    print()
    if held_out is None:
        print("  IN-SAMPLE. Every log on disk was used for training, so these")
        print("  numbers say whether the head can fit the label, not whether it")
        print("  generalises. Pass --log-names with held-out logs for that.")
    else:
        print(f"  HELD OUT: {len(held_out)} logs the model did not train on.")
    print(f"  {arguments.batches} batches of {arguments.batch_size}, "
          f"supervised cells only.\n")

    header = (
        f"  {'modality':<10}{'cells':>10}{'pearson':>9}{'spearman':>10}"
        f"{'MAE':>8}{'base':>8}{'RMSE':>8}{'base':>8}{'cal gap':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in _CHANNELS:
        row = results[name]
        if row.get("cells", 0) < 2:
            print(f"  {name:<10}{row.get('cells', 0):>10}   no supervised cells")
            continue
        print(
            f"  {name:<10}{row['cells']:>10}{row['pearson']:>9.3f}"
            f"{row['spearman']:>10.3f}{row['mae']:>8.3f}{row['baseline_mae']:>8.3f}"
            f"{row['rmse']:>8.3f}{row['baseline_rmse']:>8.3f}"
            f"{row['calibration_gap']:>9.3f}",
        )
    print(
        "\n  'base' is the same error for predicting the label mean everywhere. "
        "A head\n  that does not beat it has learned nothing usable, whatever "
        "its correlation.",
    )

    for name in _CHANNELS:
        table = results[name].get("calibration")
        if not table:
            continue
        print(f"\n  calibration, {name}:")
        print(f"    {'bin':<12}{'cells':>9}{'mean pred':>11}{'mean label':>12}")
        for low, high, cells, mean_p, mean_l in table:
            print(
                f"    {f'{low:.1f}-{high:.1f}':<12}{cells:>9}"
                f"{mean_p:>11.3f}{mean_l:>12.3f}",
            )

    if arguments.json:
        payload = {
            "model": arguments.model,
            "held_out_logs": held_out,
            "batches": arguments.batches,
            "batch_size": arguments.batch_size,
            "results": results,
        }
        pathlib.Path(arguments.json).write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
        print(f"\n  wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
