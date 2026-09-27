"""Camera destroyed: two evaluations of seed 2, against each other and the other seeds.

The spread between the two evaluations of one model is the simulator's share of
the seed-to-seed spread; what is left over is training's. Seed 1 enters by its
mean only (its CSV was lost with the old server).
"""
import csv
import pathlib
import statistics as st

R = pathlib.Path("/home/new_drive/razaghi/lead")
SEED1_CAMERA = 43.24  # thesis-artifacts/results/seed_replicates_record.md


def camera(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["modality"] == "camera" and r.get("driving_score"):
            out[r["route"]] = float(r["driving_score"])
    return out


runs = {
    "seed0": camera(R / "thesis-artifacts/results/closed_loop_diverse_dense_consistency.csv"),
    "seed2 eval1": camera(R / "results/closed_loop_diverse_dense_consistency_seed2.csv"),
    "seed2 eval2": camera(R / "results/closed_loop_diverse_dense_consistency_seed2_camera_rep2.csv"),
}
print("camera destroyed, mean DS:")
for name, d in runs.items():
    print(f"  {name:12s} {st.mean(d.values()):6.2f}  ({len(d)} routes scored)")
print(f"  {'seed1':12s} {SEED1_CAMERA:6.2f}  (recorded mean only)")

print("\npaired:")
for a, b in (("seed2 eval1", "seed2 eval2"), ("seed0", "seed2 eval1"), ("seed0", "seed2 eval2")):
    keys = sorted(set(runs[a]) & set(runs[b]))
    d = [runs[b][k] - runs[a][k] for k in keys]
    se = st.stdev(d) / len(d) ** 0.5
    print(f"  {a:12s} vs {b:12s} n={len(d):2d}  diff {st.mean(d):+6.2f}  SE {se:5.2f}  t {st.mean(d) / se:5.2f}  "
          f"identical {sum(abs(x) < 1e-9 for x in d):2d}/{len(d)}  mean |diff| {st.mean(abs(x) for x in d):5.2f}")

two = st.mean(runs["seed2 eval1"].values()), st.mean(runs["seed2 eval2"].values())
print(f"\nseed 2, average of its two evaluations: {st.mean(two):.2f}")
print("For scale: the reference checkpoint driven twice on this machine differed by 5.76 on the mean "
      "and 12.46 per route in this condition.")
