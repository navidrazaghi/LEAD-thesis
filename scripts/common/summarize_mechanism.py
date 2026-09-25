"""Turn the mechanism probe into the table the thesis argues from.

``analyze_gate.py`` writes absolute levels. Those are not the claim: what the
argument rests on is how far each model moves *away from its own clean row*
when a sensor is damaged, because the absolute attention split differs between
architectures for reasons that have nothing to do with degradation.

The direction check matters as much as the size. Damaging the camera should push
attention off the image grid and damaging the LiDAR should pull it back on. A
model that moves a lot in the wrong direction is not adapting, and reporting the
magnitude alone would hide that.
"""

import argparse
import csv
import pathlib

CLEAN = "none:0"
# Sign each condition's expected shift in camera share: away from the camera
# when the camera is the damaged one, towards it when the LiDAR is.
_EXPECTED = {"camera": -1.0, "lidar": +1.0}


def load(path: pathlib.Path) -> dict:
    """Read the probe results.

    Args:
        path: The CSV ``analyze_gate.py`` wrote.

    Returns:
        A mapping from ``(model, condition)`` to that row.
    """
    with path.open(newline="") as handle:
        return {(row["model"], row["condition"]): row for row in csv.DictReader(handle)}


def number(row: dict, field: str) -> float | None:
    """One numeric field of a row, or None when the model has no such thing.

    Args:
        row: A results row.
        field: Column to read.

    Returns:
        The value, or None if absent.
    """
    raw = (row or {}).get(field) or ""
    return float(raw) if raw not in ("", "None") else None


def cell(value: float | None, width: int = 11, digits: int = 3) -> str:
    """Format a possibly-missing number, signed, for a fixed-width table.

    Args:
        value: The number, or None.
        width: Column width.
        digits: Decimal places.

    Returns:
        The formatted cell.
    """
    return f"{'--':>{width}}" if value is None else f"{value:>+{width}.{digits}f}"


def main() -> None:
    """Print the mechanism and robustness tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("results/mechanism.csv"),
    )
    args = parser.parse_args()

    runs = load(args.csv)
    models, conditions = [], []
    for model, condition in runs:
        if model not in models:
            models.append(model)
        if condition not in conditions and condition != CLEAN:
            conditions.append(condition)

    print("=" * 78)
    print("1. MECHANISM -- shift in camera attention share vs that model's clean row")
    print("   Expected: negative when the camera is damaged, positive when the")
    print("   LiDAR is. A model with no gate has no way to shift and should sit")
    print("   near zero.")
    print("=" * 78)
    header = f"{'model':<8}{'gate':<6}" + "".join(f"{c:>12}" for c in conditions)
    print(header)

    movement: dict[str, list[float]] = {}
    for model in models:
        clean = number(runs.get((model, CLEAN)), "camera_share")
        gated = number(runs.get((model, CLEAN)), "gate_pref") is not None
        line = f"{model:<8}{'yes' if gated else 'no':<6}"
        shifts = []
        for condition in conditions:
            value = number(runs.get((model, condition)), "camera_share")
            shift = None if (value is None or clean is None) else value - clean
            if shift is not None:
                shifts.append((condition, shift))
            line += cell(shift, 12)
        print(line)
        movement[model] = shifts

    print()
    print("   direction and size, over the four damaged conditions:")
    print(f"   {'model':<8}{'mean |shift|':>14}{'correct direction':>20}")
    for model in models:
        shifts = movement.get(model) or []
        if not shifts:
            print(f"   {model:<8}{'--':>14}{'--':>20}")
            continue
        size = sum(abs(s) for _, s in shifts) / len(shifts)
        right = sum(
            1
            for condition, s in shifts
            if s * _EXPECTED[condition.split(":")[0]] > 0
        )
        print(f"   {model:<8}{size:>14.4f}{f'{right}/{len(shifts)}':>20}")

    print()
    print("=" * 78)
    print("2. ROBUSTNESS -- open-loop waypoint L2. Lower is better; * marks the")
    print("   best model in each column.")
    print("=" * 78)
    columns = [CLEAN, *conditions]
    absolute = {
        (model, condition): number(runs.get((model, condition)), "waypoint_l2")
        for model in models
        for condition in columns
    }
    best = {
        condition: min(
            (
                absolute[(m, condition)]
                for m in models
                if absolute[(m, condition)] is not None
            ),
            default=None,
        )
        for condition in columns
    }
    print(f"{'model':<8}" + "".join(f"{c:>13}" for c in columns))
    for model in models:
        line = f"{model:<8}"
        for condition in columns:
            value = absolute[(model, condition)]
            if value is None:
                line += f"{'--':>13}"
            else:
                mark = "*" if best[condition] == value else " "
                line += f"{value:>12.3f}{mark}"
        print(line)

    print()
    print("   The same numbers as a percentage of each model's own clean row.")
    print("   Read this second and read it carefully: a model that starts worse")
    print("   has less room to fall, so a small percentage here can mean a large")
    print("   absolute error above. The table above is the one that decides.")
    print(f"   {'model':<8}" + "".join(f"{c:>13}" for c in conditions))
    for model in models:
        clean = absolute[(model, CLEAN)]
        line = f"   {model:<8}"
        for condition in conditions:
            value = absolute[(model, condition)]
            if value is None or not clean:
                line += f"{'--':>13}"
            else:
                line += f"{f'{100 * (value / clean - 1):+.0f}%':>13}"
        print(line)

    print()
    print("=" * 78)
    print("3. THE CONTRAST")
    print("=" * 78)
    gated = [m for m in models if number(runs.get((m, CLEAN)), "gate_pref") is not None]
    ungated = [
        m
        for m in models
        if m not in gated and movement.get(m)
    ]

    def mean_size(names: list[str]) -> float | None:
        values = [abs(s) for name in names for _, s in movement.get(name, [])]
        return sum(values) / len(values) if values else None

    with_gate, without_gate = mean_size(gated), mean_size(ungated)
    print(f"   gated   {gated}: mean |shift| {with_gate if with_gate is None else round(with_gate, 4)}")
    print(f"   ungated {ungated}: mean |shift| {without_gate if without_gate is None else round(without_gate, 4)}")
    if with_gate and without_gate:
        print(f"   ratio: the gated models reallocate {with_gate / without_gate:.0f}x more")
        print()
        print("   The ungated models are deformable too, so the operator is not what")
        print("   reallocates attention -- they simply have no modality axis to")
        print("   shift. That is what makes this attributable to the gate rather")
        print("   than to the backbone or to the degradation curriculum.")
    print()
    print("   Note what this does NOT settle: every model carrying the gate also")
    print("   trained with the degradation curriculum, so the robustness table")
    print("   above still confounds the two. rung2a (curriculum, no gate) is the")
    print("   row that separates them.")


if __name__ == "__main__":
    main()
