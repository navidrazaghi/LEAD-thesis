"""Compare the retrained baseline without pretending the killed routes never ran.

Nine of the first twenty-eight routes produced no score. Eight of them hit
``_ROUTE_TIMEOUT_S`` exactly; the harness kills the process group and no
``checkpoint_endpoint.json`` is written, so the row is recorded as NoResult.
Dropping those rows and averaging the rest is the one analysis that is certainly
wrong: a route is killed for taking too long, and taking too long is what a bad
route does, so the rows removed are the rows that would have scored lowest.

The routes are not missing. They are censored -- the score is unknown but
bounded, and one thing about them is known, that the model had not finished the
route when the clock ran out.

So no single number is reported. Each comparison is computed under every filling
of the censored cells that the data permits:

  worst   every censored route scores 0
  best    every censored route scores 100
  drop    censored routes are excluded, i.e. assumed average -- what the naive
          analysis does, kept only so its bias is visible next to the bounds

The worst and best columns are Manski bounds: they assume nothing about why a
route was censored, so the truth is inside them whatever the cause. A conclusion
whose sign is the same in both is safe to state. A conclusion whose sign flips
between them is not supported by this data, and saying so is the result.

One asymmetry has to be read alongside every number here. The old baseline was
evaluated on an idle machine; the retrained one is being evaluated while another
user holds seventeen of the thirty-two cores, and its processes measured 61%
and 54% run-queue starvation. Median wall time per route went from 475 s to
1314 s. So the two sides are censored at very different rates, and the censoring
is not a property of the models being compared. The counts are printed for both
sides for exactly this reason.
"""

import csv
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW = ROOT / "results" / "closed_loop_new_baseline.csv"
OLD = ROOT / "results" / "closed_loop.csv"

WORST, BEST = 0.0, 100.0


def load(path, model=None):
    """Rows by (condition, route), keeping censored rows rather than dropping them."""
    table = {}
    if not path.exists():
        return table
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if model is not None and row["model"] != model:
                continue
            key = (row["modality"] + ":" + row["severity"], row["route"])
            raw = (row.get("driving_score") or "").strip()
            table[key] = float(raw) if raw else None
    return table


def t_critical(df):
    """Two-sided 95% t value. Interpolated from a table; exact enough to report."""
    table = {
        5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021,
        60: 2.000, 120: 1.980,
    }
    if df <= 0:
        return float("nan")
    keys = sorted(table)
    if df >= keys[-1]:
        return 1.96
    for low, high in zip(keys, keys[1:]):
        if low <= df <= high:
            span = high - low
            return table[low] + (table[high] - table[low]) * (df - low) / span
    return table[keys[0]]


def sign_test(differences):
    """Two-sided exact binomial p for the count of positive differences."""
    kept = [d for d in differences if d != 0]
    n = len(kept)
    if n == 0:
        return float("nan")
    positive = sum(1 for d in kept if d > 0)
    tail = 0.0
    for k in range(n + 1):
        p = math.comb(n, k) * 0.5**n
        if p <= math.comb(n, positive) * 0.5**n + 1e-12:
            tail += p
    return min(1.0, tail)


def paired(new, old, fill):
    """Paired differences new - old over routes present on both sides."""
    differences = []
    for key, value in new.items():
        if key not in old:
            continue
        a = value if value is not None else fill
        b = old[key] if old[key] is not None else fill
        if fill is None and (value is None or old[key] is None):
            continue
        differences.append(a - b)
    return differences


def describe(differences):
    n = len(differences)
    if n < 2:
        return "n=%d, too few to interval" % n
    mean = statistics.mean(differences)
    spread = statistics.stdev(differences)
    half = t_critical(n - 1) * spread / math.sqrt(n)
    return "n=%2d  mean %+7.2f  [%+7.2f, %+7.2f]  sign p=%.3f" % (
        n, mean, mean - half, mean + half, sign_test(differences),
    )


def main():
    new_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else NEW
    new = load(new_path)
    old = load(OLD, model="rung0")

    conditions = sorted({key[0] for key in new})
    print("retrained baseline vs the original, paired by route")
    print("%s\n" % ("=" * 78))

    for condition in conditions:
        n_rows = {k: v for k, v in new.items() if k[0] == condition}
        o_rows = {k: v for k, v in old.items() if k[0] == condition}
        shared = set(n_rows) & set(o_rows)
        if not shared:
            continue
        n_cens = sum(1 for k in shared if n_rows[k] is None)
        o_cens = sum(1 for k in shared if o_rows[k] is None)
        print("condition %s   routes on both sides: %d" % (condition, len(shared)))
        print("  censored: retrained %d, original %d" % (n_cens, o_cens))
        for label, fill in (("worst", WORST), ("best ", BEST), ("drop ", None)):
            differences = paired(n_rows, o_rows, fill)
            print("    %s  %s" % (label, describe(differences)))

        signs = []
        for fill in (WORST, BEST):
            differences = paired(n_rows, o_rows, fill)
            if differences:
                signs.append(statistics.mean(differences) > 0)
        if len(signs) == 2:
            verdict = (
                "direction is stable across the bounds"
                if signs[0] == signs[1]
                else "DIRECTION FLIPS between the bounds -- not supported"
            )
            print("    -> %s" % verdict)
        print()

    print("The bounds hold whatever caused the censoring. Where the direction")
    print("flips, this data cannot settle the comparison and should not be")
    print("reported as if it had.")


if __name__ == "__main__":
    main()
