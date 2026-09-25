"""Summarize the degradation pilot as a paired table.

Routes differ enormously in difficulty -- a driving score of 0 and one of 100
both appear in this pilot -- so an unpaired mean over conditions mostly measures
which routes happened to finish. Every comparison here is therefore paired on
the route, and any route missing a run under either side of a comparison is
dropped from that comparison rather than averaged around.
"""

import argparse
import collections
import csv
import pathlib
import statistics

CONDITION_ORDER = (
    "none:0",
    "camera:0.5",
    "camera:1.0",
    "lidar:0.5",
    "lidar:1.0",
)


def load(path: pathlib.Path) -> dict:
    """Read the results CSV, keeping the last row per run.

    Args:
        path: The results CSV written by ``run_evaluation.py``.

    Returns:
        A mapping from ``(model, condition, route)`` to that run's row.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = {}
    for row in rows:
        condition = f"{row['modality']}:{row['severity']}"
        kept[(row["model"], condition, row["route"])] = row
    return kept


def score(row: dict, field: str) -> float | None:
    """One numeric field of a row, or None when the run recorded nothing.

    Args:
        row: A results row.
        field: The column to read.

    Returns:
        The value as a float, or None if it is absent or empty.
    """
    raw = (row or {}).get(field) or ""
    return float(raw) if raw not in ("", "None") else None


def mean(values: list[float]) -> float | None:
    """The mean of values, or None when there are none.

    Args:
        values: The numbers to average.

    Returns:
        The arithmetic mean, or None for an empty list.
    """
    return statistics.fmean(values) if values else None


def cell(value: float | None, width: int = 7) -> str:
    """Format a possibly-missing number for a fixed-width table.

    Args:
        value: The number, or None.
        width: Column width.

    Returns:
        The formatted cell.
    """
    return f"{'--':>{width}}" if value is None else f"{value:>{width}.1f}"


def main() -> None:
    """Print the pilot summary tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("results/pilot.csv"))
    args = parser.parse_args()

    runs = load(args.csv)
    models = sorted({key[0] for key in runs})
    routes = sorted({key[2] for key in runs})
    print(f"{len(runs)} unique runs | models {models} | {len(routes)} routes\n")

    print("=" * 74)
    print("1. COVERAGE -- how many of the 10 routes each cell actually has")
    print("=" * 74)
    print(f"{'model':<8}" + "".join(f"{c:>14}" for c in CONDITION_ORDER))
    for model in models:
        counts = [
            sum(1 for r in routes if (model, c, r) in runs) for c in CONDITION_ORDER
        ]
        print(f"{model:<8}" + "".join(f"{n:>14}" for n in counts))

    print()
    print("=" * 74)
    print("2. DRIVING SCORE and ROUTE COMPLETION, paired on routes present in all")
    print("   three conditions for that model (so the curve is a curve, not a")
    print("   different route set at each point)")
    print("=" * 74)
    for model in models:
        common = [
            r for r in routes if all((model, c, r) in runs for c in CONDITION_ORDER)
        ]
        print(f"\n  {model}  (n = {len(common)} routes complete across all conditions)")
        print(f"    {'condition':<13}{'DS':>8}{'RC%':>8}{'stalled':>9}{'blocked':>9}")
        clean = None
        for condition in CONDITION_ORDER:
            rows = [runs[(model, condition, r)] for r in common]
            ds = mean([v for v in (score(x, "driving_score") for x in rows) if v is not None])
            rc = mean([v for v in (score(x, "route_completion") for x in rows) if v is not None])
            stalled = sum(1 for x in rows if "Tick" in (x["status"] or ""))
            blocked = sum(1 for x in rows if "blocked" in (x["status"] or ""))
            if condition == "none:0":
                clean = ds
            retained = ""
            if clean and ds is not None and condition != "none:0":
                retained = f"   ({100 * ds / clean:.0f}% of clean)"
            print(
                f"    {condition:<13}{cell(ds, 8)}{cell(rc, 8)}"
                f"{stalled:>9}{blocked:>9}{retained}",
            )

    print()
    print("=" * 74)
    print("3. THE COMPARISON -- rung3 minus rung0, paired route by route")
    print("   Positive favours rung3 (the observability-gated model).")
    print("=" * 74)
    if len(models) < 2:
        print("  only one model present; nothing to compare yet")
        return
    base, ours = "rung0", "rung3"
    if base not in models or ours not in models:
        base, ours = models[0], models[1]
    for condition in CONDITION_ORDER:
        pairs = [
            (score(runs[(base, condition, r)], "driving_score"),
             score(runs[(ours, condition, r)], "driving_score"), r)
            for r in routes
            if (base, condition, r) in runs and (ours, condition, r) in runs
        ]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        print(f"\n  {condition}   (n = {len(pairs)} paired routes)")
        if not pairs:
            print("    no route has both models yet")
            continue
        deltas = [b - a for a, b, _ in pairs]
        wins = sum(1 for d in deltas if d > 0)
        print(f"    {'route':<12}{base:>9}{ours:>9}{'delta':>9}")
        for a, b, r in sorted(pairs, key=lambda p: p[2]):
            print(f"    {r:<12}{a:>9.1f}{b:>9.1f}{b - a:>+9.1f}")
        spread = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        print(f"    {'mean':<12}{mean([a for a, _, _ in pairs]):>9.1f}"
              f"{mean([b for _, b, _ in pairs]):>9.1f}{mean(deltas):>+9.1f}")
        print(f"    {ours} better on {wins}/{len(pairs)} routes; "
              f"per-route sd of the delta {spread:.1f}")
    print(
        "\n  Read the sd next to the mean before claiming anything: with ten "
        "routes\n  and per-route spread this wide, only a large effect is "
        "separable from noise.",
    )


if __name__ == "__main__":
    main()
