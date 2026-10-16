"""Pair the corrected rung 2a against the corrected baseline, route by route.

Averages over two route sets are not comparable when either set is missing
rows: a route the harness killed carries no score, and it is not missing at
random -- a route is killed for taking too long, which is what a hard route
does. So the comparison here is per route, and a route drops out unless both
models scored it. How many dropped, and from which side, is printed rather
than folded away.

The sign test is exact and two-sided: under the null that a route is equally
likely to go either way, the probability of a split at least this lopsided.
It asks only about direction, which is the question a paired comparison over
thirty routes can actually answer.
"""

import collections
import csv
import math
import pathlib
import statistics
import sys

BASE = pathlib.Path.home() / "LEAD/lead/results"
CONDITIONS = [("none", "0"), ("lidar", "1.0"), ("camera", "1.0")]
LABEL = {"none": "intact", "lidar": "LiDAR destroyed", "camera": "camera destroyed"}


def load(name: str) -> tuple[dict, set]:
    """Read one campaign.

    Args:
        name: File name under results/.

    Returns:
        Scores keyed by (modality, severity, route), and the set of keys that
        appeared but carried no score.
    """
    scores, censored = {}, set()
    with (BASE / name).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["modality"], row["severity"], row["route"])
            raw = (row.get("driving_score") or "").strip()
            try:
                scores[key] = float(raw)
            except ValueError:
                censored.add(key)
    return scores, censored


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test.

    Args:
        better: Routes where the treatment scored higher.
        worse: Routes where it scored lower.

    Returns:
        The p-value, or 1.0 when there is nothing to test.
    """
    n = better + worse
    if n == 0:
        return 1.0
    extreme = min(better, worse)
    tail = sum(math.comb(n, k) for k in range(extreme + 1))
    return min(1.0, 2 * tail / 2 ** n)


treated, treated_censored = load("closed_loop_rung2a_recipe.csv")
control, control_censored = load("closed_loop_post31.csv")

print("rung2a_recipe (dense + degradation curriculum)  vs  post31 (baseline)")
print("both at the published recipe: 31 epochs, batch 32 x accum 2, same 450 logs")

for modality, severity in CONDITIONS:
    routes = {r for (m, s, r) in list(treated) + list(control)
              if (m, s) == (modality, severity)}
    pairs, dropped = [], []
    for route in sorted(routes):
        key = (modality, severity, route)
        if key in treated and key in control:
            pairs.append((route, treated[key] - control[key],
                          treated[key], control[key]))
        else:
            side = []
            if key not in treated:
                side.append("rung2a_recipe")
            if key not in control:
                side.append("post31")
            dropped.append((route, "+".join(side)))

    deltas = [d for _, d, _, _ in pairs]
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)
    tied = sum(1 for d in deltas if d == 0)

    print(f"\n=== {LABEL[modality]} ===")
    print(f"  paired routes        {len(pairs)}")
    if dropped:
        for route, side in dropped:
            print(f"  DROPPED {route}: no score from {side}")
    if not pairs:
        continue
    print(f"  mean difference      {statistics.fmean(deltas):+7.2f}")
    print(f"  median difference    {statistics.median(deltas):+7.2f}")
    if len(deltas) > 1:
        spread = statistics.stdev(deltas)
        print(f"  std of differences   {spread:7.2f}"
              f"   (standard error {spread / math.sqrt(len(deltas)):.2f})")
    print(f"  routes better/worse/tied   {better} / {worse} / {tied}")
    print(f"  sign test p          {sign_test(better, worse):.4f}")

    ranked = sorted(pairs, key=lambda row: row[1])
    print("  three worst routes:")
    for route, delta, a, b in ranked[:3]:
        print(f"    {route:<14} {delta:+7.2f}   ({a:6.2f} vs {b:6.2f})")
    print("  three best routes:")
    for route, delta, a, b in ranked[-3:][::-1]:
        print(f"    {route:<14} {delta:+7.2f}   ({a:6.2f} vs {b:6.2f})")

print("\n=== censored rows, by campaign ===")
for name, censored in (("rung2a_recipe", treated_censored), ("post31", control_censored)):
    print(f"  {name:<14} {len(censored)}")
    for key in sorted(censored):
        print(f"      {key[0]}:{key[1]}  {key[2]}")
