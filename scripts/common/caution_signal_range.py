"""How far the caution signal actually moves, from already-recorded runs.

Before spending a GPU night on a governor, it is worth knowing whether the
signal driving it moves at all under the conditions it will be scored on. This
answers that from ``results/mechanism.csv``, which already holds the
observability head's mean output per modality for every rung and condition, so
it costs nothing to run.

The answer for the default rule is that it barely moves, and that is not a bug:
with one modality destroyed the other still resolves the scene, and the trained
rungs drive no worse under full camera destruction than intact. A governor that
slowed down there would be buying nothing with route completion. What the number
does tell you is which condition the governor has to be evaluated on -- joint
degradation, where redundancy has nothing left to fall back on.

One caveat the output repeats: this file records the mean over cells per
modality, while the governor combines per cell and then averages. For the
"best" rule the recorded number is therefore a lower bound on resolvedness, so
the caution printed here is an upper bound. The conclusion survives it: an upper
bound that small is still small.
"""

import argparse
import csv
import pathlib
import sys

_RULES = {
    "best": max,
    "mean": lambda camera, lidar: 0.5 * (camera + lidar),
    "worst": min,
}


def main() -> int:
    """Print the caution range per rung and rule.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mechanism",
        type=pathlib.Path,
        default=pathlib.Path("results/mechanism.csv"),
        help="Recorded per-modality observability means.",
    )
    arguments = parser.parse_args()

    if not arguments.mechanism.exists():
        print(f"No such file: {arguments.mechanism}", file=sys.stderr)
        return 1

    with arguments.mechanism.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["obs_camera"].strip()]
    if not rows:
        print("No rows carry an observability measurement.", file=sys.stderr)
        return 1

    models = sorted({row["model"] for row in rows})
    print("Caution = 1 - combined resolvedness, from the recorded head means.")
    print("For the 'best' rule these are upper bounds; see the module docstring.\n")

    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        print(f"{model}")
        header = f"  {'condition':<14}" + "".join(f"{rule:>9}" for rule in _RULES)
        print(header)
        for row in model_rows:
            camera = float(row["obs_camera"])
            lidar = float(row["obs_lidar"])
            cautions = "".join(
                f"{1.0 - combine(camera, lidar):9.3f}" for combine in _RULES.values()
            )
            print(f"  {row['condition']:<14}{cautions}")

        intact = next((r for r in model_rows if r["condition"] == "none:0"), None)
        damaged = next(
            (r for r in model_rows if r["condition"] == "camera:1.0"),
            None,
        )
        if intact and damaged:
            print(f"  {'swing':<14}", end="")
            for combine in _RULES.values():
                base = 1.0 - combine(
                    float(intact["obs_camera"]), float(intact["obs_lidar"]),
                )
                hit = 1.0 - combine(
                    float(damaged["obs_camera"]), float(damaged["obs_lidar"]),
                )
                print(f"{hit - base:+9.3f}", end="")
            print()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
