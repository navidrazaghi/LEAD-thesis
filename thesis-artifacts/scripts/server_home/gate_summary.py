"""Learned gate, oracle gate and no gate, route by route on the 30 routes.

Three comparisons, each paired by route:

* learned gate vs model 3 (no gate): did adding the gate change driving? The two
  differ only in the gate, but they are two trainings, so seed variability is in
  the difference (the seed-aware detectable difference on this set is ~21 points).
* oracle vs learned gate: the same checkpoint driven twice, only the gate's
  input differing. No training noise; what is left is the estimator's effect
  plus closed-loop run-to-run noise (about 6 points on the mean under camera
  destruction, measured on the reference checkpoint).
* oracle vs model 3: does reweighting by the true reliability beat not gating?
"""
import csv
import pathlib
import statistics as st

R = pathlib.Path("/home/new_drive/razaghi/lead")
FILES = {
    "baseline": "thesis-artifacts/results/closed_loop_diverse_baseline.csv",
    "model 3 (no gate)": "thesis-artifacts/results/closed_loop_diverse_curriculum2.csv",
    "learned gate": "results/closed_loop_diverse_gate.csv",
    "oracle gate": "results/closed_loop_diverse_gate_oracle.csv",
}
CONDITIONS = ("none", "lidar", "camera")


def load(path):
    out = {}
    p = R / path
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        if r.get("driving_score"):
            out[(r["modality"], r["route"])] = float(r["driving_score"])
    return out


data = {name: load(path) for name, path in FILES.items()}
print(f"{'model':20s}" + "".join(f"{c:>12s}" for c in CONDITIONS))
for name, d in data.items():
    cells = []
    for c in CONDITIONS:
        v = [x for (m, _), x in d.items() if m == c]
        cells.append(f"{st.mean(v):7.2f} ({len(v):2d})" if v else f"{'-':>12s}")
    print(f"{name:20s}" + "".join(f"{x:>12s}" for x in cells))


def paired(a, b, cond):
    keys = sorted(k for k in data[a] if k[0] == cond and k in data[b])
    if len(keys) < 2:
        return None
    d = [data[a][k] - data[b][k] for k in keys]
    se = st.stdev(d) / len(d) ** 0.5
    return st.mean(d), se, len(d), sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)


print()
for a, b in (("learned gate", "model 3 (no gate)"), ("oracle gate", "learned gate"),
             ("oracle gate", "model 3 (no gate)"), ("learned gate", "baseline"), ("oracle gate", "baseline")):
    for cond in CONDITIONS:
        r = paired(a, b, cond)
        if r is None:
            continue
        mean, se, n, w, t, l = r
        tval = mean / se if se > 0 else float("nan")
        print(f"{a:>13s} - {b:<18s} {cond:7s} {mean:+7.2f}  SE {se:5.2f}  t {tval:6.2f}  "
              f"better/equal/worse {w}/{t}/{l}  n={n}")
