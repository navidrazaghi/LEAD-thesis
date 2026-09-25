"""Turn the closed-loop sweep into the table the thesis concludes from.

Driving score varies enormously between routes -- the pilot measured a
per-route standard deviation of about 31 against a mean near 30, with
individual routes scoring anything from 0 to 100. Two consequences follow, and
this script is built around both.

First, every comparison is **paired on the route**. An unpaired mean over
models mostly measures which routes each model happened to get through.

Second, a mean difference is reported with the spread of the differences beside
it. With this much noise a five-point gap between two models can easily be
nothing, and a table that prints only the means invites reading it as a
finding. The interval printed here is roughly 95% (mean +/- 2 standard errors);
when it straddles zero, the honest statement is that the sweep did not resolve
a difference, not that the models are equal.

Rows whose status names a simulator failure carry no measurement and are
dropped. A TickRuntime is *not* such a status -- the agent burned its whole
step budget without finishing, which is bad driving and scores accordingly.
"""

import argparse
import collections
import csv
import math
import pathlib
import statistics

_INFRASTRUCTURE = ("NoResult", "Agent timed out")
# n is 30 by design, where the two-sided 95% t value is 2.045; 2.0 is close
# enough for a spread this large and does not pretend to more precision.
_INTERVAL = 2.0


def usable(row: dict) -> bool:
    """Whether a row carries a driving measurement.

    Args:
        row: A results row.

    Returns:
        True when the row can be averaged.
    """
    status = row.get("status") or ""
    score = row.get("driving_score") or ""
    if score in ("", "None"):
        return False
    return not any(marker in status for marker in _INFRASTRUCTURE)


def load(path: pathlib.Path) -> dict:
    """Read the sweep results, keeping the last row per run.

    A resumed sweep re-runs anything that did not score, so the same run can
    appear twice; the newer attempt is the one to keep.

    Args:
        path: The results CSV.

    Returns:
        A mapping from ``(model, condition, route)`` to that row.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kept = {}
    for row in rows:
        condition = f"{row['modality']}:{row['severity']}"
        kept[(row["model"], condition, row["route"])] = row
    return kept


def value(row: dict | None, field: str) -> float | None:
    """One numeric field of a row, or None when it is absent.

    Args:
        row: A results row.
        field: Column to read.

    Returns:
        The value as a float, or None.
    """
    raw = (row or {}).get(field) or ""
    return float(raw) if raw not in ("", "None") else None


def paired(runs: dict, models: list[str], condition: str, routes: list[str]) -> list[str]:
    """The routes every model has a usable measurement for under one condition.

    Args:
        runs: The loaded results.
        models: Model names.
        condition: The condition to look at.
        routes: All routes seen.

    Returns:
        The routes usable for a paired comparison.
    """
    return [
        route
        for route in routes
        if all(
            (model, condition, route) in runs
            and usable(runs[(model, condition, route)])
            for model in models
        )
    ]


def main() -> None:
    """Print the coverage, the scores, and the paired comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("results/closed_loop.csv"),
    )
    parser.add_argument(
        "--reference",
        default="rung2a",
        help="Model the others are compared against; the rival explanation.",
    )
    args = parser.parse_args()

    runs = load(args.csv)
    models, conditions, routes = [], [], []
    for model, condition, route in runs:
        if model not in models:
            models.append(model)
        if condition not in conditions:
            conditions.append(condition)
        if route not in routes:
            routes.append(route)
    routes.sort()

    print(f"{len(runs)} runs | models {models} | {len(routes)} routes\n")

    print("=" * 76)
    print("1. COVERAGE -- usable runs per cell, and what the rest were")
    print("=" * 76)
    print(f"{'model':<9}" + "".join(f"{c:>14}" for c in conditions))
    for model in models:
        line = f"{model:<9}"
        for condition in conditions:
            have = sum(
                1
                for route in routes
                if (model, condition, route) in runs
                and usable(runs[(model, condition, route)])
            )
            line += f"{f'{have}/{len(routes)}':>14}"
        print(line)
    statuses = collections.Counter(
        (row.get("status") or "unknown") for row in runs.values()
    )
    print("\n   statuses across every run:")
    for status, count in statuses.most_common():
        print(f"     {count:>4}  {status}")

    print()
    print("=" * 76)
    print("2. DRIVING SCORE, paired on the routes every model completed")
    print("=" * 76)
    for condition in conditions:
        common = paired(runs, models, condition, routes)
        print(f"\n  {condition}   (n = {len(common)} routes usable for all models)")
        if not common:
            print("    nothing paired yet")
            continue
        print(f"    {'model':<9}{'DS':>8}{'RC%':>8}{'blocked':>9}{'stalled':>9}")
        for model in models:
            rows = [runs[(model, condition, route)] for route in common]
            score = statistics.fmean(
                [v for v in (value(r, "driving_score") for r in rows) if v is not None],
            )
            completion = [
                v for v in (value(r, "route_completion") for r in rows) if v is not None
            ]
            blocked = sum(1 for r in rows if "blocked" in (r.get("status") or ""))
            stalled = sum(1 for r in rows if "Tick" in (r.get("status") or ""))
            done = f"{statistics.fmean(completion):>8.1f}" if completion else f"{'--':>8}"
            print(f"    {model:<9}{score:>8.1f}{done}{blocked:>9}{stalled:>9}")

    print()
    print("=" * 76)
    print(f"3. THE COMPARISON -- each model minus {args.reference}, paired route by route")
    print("=" * 76)
    if args.reference not in models:
        print(f"  reference {args.reference!r} is not in the results; nothing to compare")
        return
    for condition in conditions:
        common = paired(runs, models, condition, routes)
        print(f"\n  {condition}   (n = {len(common)})")
        if len(common) < 2:
            print("    too few paired routes")
            continue
        for model in models:
            if model == args.reference:
                continue
            deltas = [
                value(runs[(model, condition, route)], "driving_score")
                - value(runs[(args.reference, condition, route)], "driving_score")
                for route in common
            ]
            mean = statistics.fmean(deltas)
            spread = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            error = spread / math.sqrt(len(deltas)) if deltas else 0.0
            low, high = mean - _INTERVAL * error, mean + _INTERVAL * error
            wins = sum(1 for d in deltas if d > 0)
            verdict = (
                "resolved" if low > 0 or high < 0 else "NOT resolved (interval spans 0)"
            )
            print(
                f"    {model:<9} vs {args.reference:<9} "
                f"mean {mean:+6.1f}  sd {spread:5.1f}  se {error:5.1f}  "
                f"95% [{low:+6.1f}, {high:+6.1f}]  better on {wins}/{len(deltas)}  {verdict}",
            )
    print(
        "\n  A positive mean favours the first model. Read the interval before the\n"
        "  mean: with a per-route spread this wide, only an interval clear of zero\n"
        "  supports a claim.",
    )


if __name__ == "__main__":
    main()
