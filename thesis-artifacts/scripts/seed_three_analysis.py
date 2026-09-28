# -*- coding: utf-8 -*-
"""Three seeds of dense + curriculum v2 + consistency, and what they change.

Seeds 0 and 1 ran on the original server; seed 2 on its replacement, with the
same 585 logs, recipe and 30 routes. Seed 1's CSV was lost with the first
machine, so its means and its paired gaps to seed 0 are taken from
results/seed_replicates_record.md, which transcribes the server's output.

The minimum-detectable-difference method is the one the thesis already uses
(chapter 5): the route-only threshold from the median spread of the campaign's
paired comparisons, and a per-run standard deviation from the seed replicate,
combined as sqrt(mdd_routes^2 + 2 sd^2). With two seeds the per-run sd was
sqrt(mean(gap^2) / 2); its three-seed form is the pooled sample variance of the
seed means, which reduces to the same thing when there are two. The script
reproduces the two-seed numbers first, as a check that it is the same method.

Usage: python seed_three_analysis.py   (from thesis-artifacts/scripts)
"""
import csv
import pathlib
import statistics as st

import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent / "results"
CONDITIONS = ("none", "lidar", "camera")


def load(name, model=None):
    out = {}
    for row in csv.DictReader(open(R / name, encoding="utf-8")):
        if row.get("driving_score") and (model is None or row["model"] == model):
            out[(row["modality"], row["route"])] = float(row["driving_score"])
    return out


def paired(a, b, cond):
    keys = sorted(k for k in a if k[0] == cond and k in b)
    d = [b[k] - a[k] for k in keys]
    se = st.stdev(d) / len(d) ** 0.5
    return {"n": len(d), "mean": st.mean(d), "se": se, "t": st.mean(d) / se,
            "abs": st.mean(abs(x) for x in d), "same": sum(abs(x) < 1e-9 for x in d)}


def mean_of(d, cond):
    v = [x for (m, _), x in d.items() if m == cond]
    return st.mean(v), len(v)


baseline = load("closed_loop_diverse_baseline.csv")
deformable = load("closed_loop_diverse_deformable.csv")
deformable_v2 = load("closed_loop_diverse_curriculum2.csv")
dense_v2 = load("closed_loop_diverse_dense_curriculum2.csv")
seed0 = load("closed_loop_diverse_dense_consistency.csv")
seed2 = load("closed_loop_diverse_dense_consistency_seed2.csv")
seed2_rep = load("closed_loop_diverse_dense_consistency_seed2_camera_rep2.csv")
ref1 = load("closed_loop_reference_v150.csv")
ref2 = load("closed_loop_reference_v150_camera_rep2.csv")
SEED1_MEANS = {"none": 71.96, "lidar": 69.59, "camera": 43.24}      # record, seed 1
SEED1_GAPS = np.array([14.48, 9.21, -14.38])                        # record, seed 1 - seed 0

print("1. SEED MEANS")
means = {}
for c in CONDITIONS:
    m0, n0 = mean_of(seed0, c)
    m2, n2 = mean_of(seed2, c)
    mb, _ = mean_of(baseline, c)
    trio = [m0, SEED1_MEANS[c], m2]
    means[c] = trio
    print(f"  {c:7s} seed0 {m0:6.2f} ({n0})  seed1 {SEED1_MEANS[c]:6.2f}  seed2 {m2:6.2f} ({n2})"
          f"  mean {st.mean(trio):6.2f}  sd {st.stdev(trio):5.2f}  baseline {mb:6.2f}"
          f"  mean-baseline {st.mean(trio) - mb:+6.2f}")

print("\n2. PAIRED, SEED 2")
for label, a, b in (("seed2 - seed0", seed0, seed2), ("seed2 - baseline", baseline, seed2),
                    ("seed2 - dense+v2", dense_v2, seed2)):
    for c in CONDITIONS:
        r = paired(a, b, c)
        print(f"  {label:17s} {c:7s} n={r['n']:2d} {r['mean']:+7.2f} SE {r['se']:5.2f} t {r['t']:5.2f}")

print("\n3. THE SAME CHECKPOINT DRIVEN TWICE (camera destroyed)")
eval_gaps = []
for label, a, b in (("reference v150", ref1, ref2), ("seed 2", seed2, seed2_rep)):
    r = paired(a, b, "camera")
    eval_gaps.append(r["mean"])
    print(f"  {label:15s} n={r['n']} {r['mean']:+6.2f} SE {r['se']:4.2f}  per-route |diff| {r['abs']:5.2f}"
          f"  identical {r['same']}/{r['n']}")
sd_eval_mean = np.sqrt((np.array(eval_gaps) ** 2).mean() / 2)
print(f"  implied sd of one evaluation's mean: {sd_eval_mean:.2f}")

print("\n4. MINIMUM DETECTABLE DIFFERENCE")
models = {"baseline": baseline, "deformable": deformable, "deformable+v2": deformable_v2,
          "dense+v2": dense_v2, "dense+v2+consistency": seed0}
rows = []
for b, a in (("deformable", "baseline"), ("deformable+v2", "deformable"), ("dense+v2", "baseline"),
             ("dense+v2+consistency", "dense+v2"), ("dense+v2", "deformable+v2")):
    for c in CONDITIONS:
        rows.append(paired(models[a], models[b], c))
sd_routes = np.median([r["se"] * np.sqrt(r["n"]) for r in rows])
n = 30
mdd_routes = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * sd_routes / np.sqrt(n)
sd_two = np.sqrt((SEED1_GAPS ** 2).mean() / 2)
sd_three = np.sqrt(np.mean([st.variance(means[c]) for c in CONDITIONS]))
mdd_two = np.sqrt(mdd_routes ** 2 + 2 * sd_two ** 2)
mdd_three = np.sqrt(mdd_routes ** 2 + 2 * sd_three ** 2)
print(f"  route-only threshold: {mdd_routes:.1f}   (thesis: 16.2)")
print(f"  two seeds:   per-run sd {sd_two:.1f}, threshold {mdd_two:.1f}   (thesis: 9.1, 20.7)")
print(f"  three seeds: per-run sd {sd_three:.1f}, threshold {mdd_three:.1f}")
print(f"  of the per-run variance, evaluation noise on the mean accounts for "
      f"{sd_eval_mean ** 2 / sd_three ** 2:.0%}")
