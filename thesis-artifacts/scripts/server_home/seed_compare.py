"""Dense + v2 + consistency across seeds 0, 1 and 2.

Seeds 0 and 2 have per-route CSVs, so they pair route by route. Seed 1's CSV
was lost with the old server; only its means survive, transcribed in
thesis-artifacts/results/seed_replicates_record.md, so it enters as means only.
Seed 2 was trained and scored on the new server, the others on the old one --
the system tests found no difference between the two machines beyond ordinary
run-to-run noise, which in the camera-destroyed condition is large (two runs of
one checkpoint differed by 12.5 points per route on average).
"""
import csv
import pathlib
import statistics as st

R = pathlib.Path("/home/new_drive/razaghi/lead")
SEED1 = {"none": 71.96, "lidar": 69.59, "camera": 43.24}  # seed_replicates_record.md
BASELINE = R / "thesis-artifacts/results/closed_loop_diverse_baseline.csv"


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get("driving_score"):
            out[(r["modality"], r["route"])] = float(r["driving_score"])
    return out


s0 = load(R / "thesis-artifacts/results/closed_loop_diverse_dense_consistency.csv")
s2 = load(R / "results/closed_loop_diverse_dense_consistency_seed2.csv")
base = load(BASELINE)

print(f"{'condition':10s} {'seed 0':>7s} {'seed 1':>7s} {'seed 2':>7s} {'3-seed mean':>12s} {'sd':>6s} {'baseline':>9s}")
for cond in ("none", "lidar", "camera"):
    m0 = st.mean(v for (m, _), v in s0.items() if m == cond)
    m2 = st.mean(v for (m, _), v in s2.items() if m == cond)
    mb = st.mean(v for (m, _), v in base.items() if m == cond)
    three = [m0, SEED1[cond], m2]
    print(f"{cond:10s} {m0:7.2f} {SEED1[cond]:7.2f} {m2:7.2f} {st.mean(three):12.2f} {st.stdev(three):6.2f} {mb:9.2f}")

print("\nseed 2 against seed 0, paired by route:")
for cond in ("none", "lidar", "camera"):
    keys = [k for k in s2 if k[0] == cond and k in s0]
    d = [s2[k] - s0[k] for k in keys]
    se = st.stdev(d) / len(d) ** 0.5
    print(f"  {cond:7s} n={len(d):2d}  diff {st.mean(d):+6.2f}  SE {se:5.2f}  t {st.mean(d) / se:5.2f}")

print("\nseed 2 against the baseline (seed 0), paired by route:")
for cond in ("none", "lidar", "camera"):
    keys = [k for k in s2 if k[0] == cond and k in base]
    d = [s2[k] - base[k] for k in keys]
    se = st.stdev(d) / len(d) ** 0.5
    print(f"  {cond:7s} n={len(d):2d}  diff {st.mean(d):+6.2f}  SE {se:5.2f}  t {st.mean(d) / se:5.2f}")
