"""Decide which checkpoints earn a closed-loop night -- and whether to trust that.

A closed-loop night costs about eleven hours per model per condition, and the
open-loop error over cached frames costs minutes. Using the cheap number to
choose which models get the expensive one is only sound if the cheap number
ranks models the way the expensive one does. That is an empirical claim about
this stack, not a general property of open-loop evaluation, and it is checkable
against runs that already exist.

So this has two jobs and keeps them apart. ``validate`` compares recorded
open-loop error against recorded closed-loop driving score and reports whether
the ranking agrees. ``screen`` ranks a set of candidates by open-loop error, and
refuses to pretend that ranking means anything the validation has not earned.

The distinction matters because the failure mode is silent. A screen that does
not predict will still produce a confident ordering, and the models it discards
never get the night that would have shown it was wrong.
"""

import argparse
import collections
import csv
import itertools
import pathlib
import statistics
import sys

# Conditions are named the same way in both files, which is what lets them be
# joined at all: modality:severity, as the evaluation sweep spells it.
_INTACT = "none:0"


def read_openloop(path: pathlib.Path) -> dict[tuple[str, str], float]:
    """Mean per-frame waypoint error, keyed by model and condition.

    Args:
        path: Per-frame CSV with model, condition and l2 columns.

    Returns:
        The mean error of each ``(model, condition)`` cell.
    """
    frames: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frames[(row["model"], row["condition"])].append(float(row["l2"]))
    return {key: statistics.mean(values) for key, values in frames.items()}


def read_closedloop(path: pathlib.Path) -> dict[tuple[str, str], float]:
    """Mean driving score, keyed by model and condition.

    Args:
        path: The closed-loop CSV the evaluation sweep appends to.

    Returns:
        The mean driving score of each ``(model, condition)`` cell, over the
        routes that produced a score.
    """
    scores: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["driving_score"].strip():
                continue
            severity = row["severity"]
            condition = f"{row['modality']}:{severity}"
            scores[(row["model"], condition)].append(float(row["driving_score"]))
    return {key: statistics.mean(values) for key, values in scores.items()}


def spearman(first: list[float], second: list[float]) -> float:
    """Rank correlation of two equal-length sequences.

    Computed here rather than imported so this script runs anywhere the
    evaluation CSVs do, and because with ties the definition worth using is the
    one over average ranks.

    Args:
        first: One sequence.
        second: The other, same length.

    Returns:
        The correlation in ``[-1, 1]``; zero when either side is constant.
    """
    if len(first) < 2:
        return 0.0
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    mean_first = statistics.mean(first_ranks)
    mean_second = statistics.mean(second_ranks)
    covariance = sum(
        (a - mean_first) * (b - mean_second)
        for a, b in zip(first_ranks, second_ranks, strict=True)
    )
    spread_first = sum((a - mean_first) ** 2 for a in first_ranks)
    spread_second = sum((b - mean_second) ** 2 for b in second_ranks)
    if spread_first == 0.0 or spread_second == 0.0:
        return 0.0
    return covariance / (spread_first * spread_second) ** 0.5


def _average_ranks(values: list[float]) -> list[float]:
    """Rank values from smallest to largest, ties sharing their average rank.

    Args:
        values: The values to rank.

    Returns:
        One rank per value, in the input order.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[position]]:
            stop += 1
        shared = (position + stop) / 2.0 + 1.0
        for index in order[position : stop + 1]:
            ranks[index] = shared
        position = stop + 1
    return ranks


def validate(openloop: dict, closedloop: dict) -> int:
    """Report whether open-loop error ranks models the way driving score does.

    Args:
        openloop: Mean open-loop error per cell.
        closedloop: Mean driving score per cell.

    Returns:
        Process exit code; non-zero when the two disagree badly enough that the
        screen should not be used to spend nights.
    """
    shared = sorted(set(openloop) & set(closedloop))
    if len(shared) < 3:
        print(
            f"Only {len(shared)} cell(s) have both measurements; nothing to "
            f"validate against.",
            file=sys.stderr,
        )
        return 1

    print(f"Cells with both measurements: {len(shared)}\n")
    print(f"  {'model':<8} {'condition':<12} {'open-loop L2':>13} {'driving score':>14}")
    for model, condition in shared:
        print(
            f"  {model:<8} {condition:<12} "
            f"{openloop[(model, condition)]:13.4f} "
            f"{closedloop[(model, condition)]:14.1f}",
        )

    errors = [openloop[key] for key in shared]
    scores = [closedloop[key] for key in shared]
    # Negated because lower error is meant to mean better driving: a screen that
    # works gives a positive number here.
    pooled = -spearman(errors, scores)
    print(f"\nPooled rank correlation (lower error vs higher score): {pooled:+.3f}")

    # Pooled agreement is the easy test and it is not the one that matters. The
    # screen is used to choose between models *within* a condition, and a strong
    # pooled number can come entirely from conditions differing from each other
    # while the models inside each are ordered wrongly.
    print("\nWithin-condition ordering, which is what the screen is used for:")
    conditions = sorted({condition for _, condition in shared})
    agreements = []
    for condition in conditions:
        cells = [key for key in shared if key[1] == condition]
        if len(cells) < 2:
            continue
        pairs = 0
        agreed = 0
        for left, right in itertools.combinations(cells, 2):
            if openloop[left] == openloop[right]:
                continue
            pairs += 1
            screen_prefers_left = openloop[left] < openloop[right]
            truth_prefers_left = closedloop[left] > closedloop[right]
            agreed += int(screen_prefers_left == truth_prefers_left)
        if not pairs:
            continue
        agreements.append(agreed / pairs)
        verdict = "agrees" if agreed == pairs else "DISAGREES"
        print(f"  {condition:<12} {agreed}/{pairs} pairs {verdict}")
        for left, right in itertools.combinations(cells, 2):
            screen_prefers_left = openloop[left] < openloop[right]
            truth_prefers_left = closedloop[left] > closedloop[right]
            if screen_prefers_left != truth_prefers_left:
                better, worse = (left, right) if screen_prefers_left else (right, left)
                print(
                    f"      screen picks {better[0]} over {worse[0]}, "
                    f"but closed loop scores {closedloop[better]:.1f} "
                    f"against {closedloop[worse]:.1f}",
                )

    if not agreements:
        print("  no condition had two models to compare.")
        return 1

    overall = statistics.mean(agreements)
    print(f"\nWithin-condition pair agreement: {overall:.0%}")
    if overall < 0.75:
        print(
            "\nVERDICT: the screen does not rank models the way the simulator "
            "does.\nUsing it to allocate closed-loop nights would discard "
            "models on a signal\nthat has been measured not to predict the "
            "outcome. Run the grid instead.",
        )
        return 2
    print("\nVERDICT: the screen's ordering is consistent with the simulator's.")
    return 0


def screen(openloop: dict, conditions: list[str]) -> int:
    """Rank candidates by open-loop error within each condition.

    Args:
        openloop: Mean open-loop error per cell.
        conditions: Conditions to rank within; empty ranks within all of them.

    Returns:
        Process exit code.
    """
    wanted = set(conditions) if conditions else {key[1] for key in openloop}
    for condition in sorted(wanted):
        cells = sorted(
            (key for key in openloop if key[1] == condition),
            key=lambda key: openloop[key],
        )
        if not cells:
            continue
        print(f"\n{condition}")
        for rank, key in enumerate(cells, start=1):
            marker = ""
            if condition != _INTACT and (key[0], _INTACT) in openloop:
                cost = openloop[key] - openloop[(key[0], _INTACT)]
                marker = f"   (+{cost:.4f} over intact)"
            print(f"  {rank}. {key[0]:<10} {openloop[key]:.4f}{marker}")
    print(
        "\nThis ordering is only as good as the validation subcommand says it "
        "is.\nCheck that before spending a night on it.",
    )
    return 0


def main() -> int:
    """Run the requested subcommand.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "screen"))
    parser.add_argument(
        "--openloop",
        type=pathlib.Path,
        default=pathlib.Path("results/openloop_frames.csv"),
        help="Per-frame open-loop errors.",
    )
    parser.add_argument(
        "--closedloop",
        type=pathlib.Path,
        default=pathlib.Path("results/closed_loop.csv"),
        help="Closed-loop driving scores, for validation.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=[],
        help="Conditions to rank within; default is all present.",
    )
    arguments = parser.parse_args()

    if not arguments.openloop.exists():
        print(f"No such file: {arguments.openloop}", file=sys.stderr)
        return 1
    openloop = read_openloop(arguments.openloop)

    if arguments.mode == "screen":
        return screen(openloop, arguments.conditions)

    if not arguments.closedloop.exists():
        print(f"No such file: {arguments.closedloop}", file=sys.stderr)
        return 1
    return validate(openloop, read_closedloop(arguments.closedloop))


if __name__ == "__main__":
    sys.exit(main())
