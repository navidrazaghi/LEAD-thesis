"""Which catalogue faults break which model, and how badly.

Two checkpoints go through the same sweep: the seed-2 consistency model trained
on our 585 logs, and the published LEAD checkpoint trained on the full 1.1 TB
release. Each fault is read against the same checkpoint driven intact on the
same routes, route by route, so a drop is the fault's and not the route mix's.
The thesis's own conditions (camera:1.0, lidar:1.0) are listed for scale.

The side-by-side table then asks the question the reference is here for: does a
model trained on twenty times more data degrade less under the same fault? Its
last column is the paired difference between the two models' drops on the same
routes.

Usage: fault_probe_summary.py SWEEP_CSV ROUTES_FILE
"""
import csv
import pathlib
import statistics as st
import sys

R = pathlib.Path("/home/new_drive/razaghi/lead")
sweep_path, routes_path = sys.argv[1], sys.argv[2]
routes = set(pathlib.Path(routes_path).read_text().split())

MODELS = {
    "dense_consistency_seed2": ("seed 2 (585 logs)", "results/closed_loop_diverse_dense_consistency_seed2.csv"),
    "reference_v150": ("LEAD reference (1.1 TB)", "results/closed_loop_reference_v150.csv"),
}


def load(path, model):
    out = {}
    path = pathlib.Path(path)
    path = path if path.is_absolute() else R / path
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        if r["model"] == model and r["route"] in routes and r.get("driving_score"):
            out[(r["modality"], float(r["severity"] or 0), r["route"])] = float(r["driving_score"])
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
    return st.mean(d), se


drops = {}
for model, (label, own_csv) in MODELS.items():
    own = load(own_csv, model)
    sweep = load(sweep_path, model)
    intact = {k[2]: v for k, v in own.items() if k[0] == "none"}
    if not intact:
        print(f"{label}: no intact rows on these routes\n")
        continue
    print(f"=== {label}: intact DS {st.mean(intact.values()):.2f} on {len(intact)} routes")
    print(f"{'condition':26s} {'n':>3s} {'DS':>6s} {'drop':>7s} {'SE':>5s} {'t':>6s}")
    rows = []
    for source, thesis in ((own, True), (sweep, False)):
        for cond in sorted({k[:2] for k in source if k[0] != "none"}):
            keys = [k for k in source if k[:2] == cond and k[2] in intact]
            if len(keys) < 2:
                continue
            ds = [source[k] for k in keys]
            mean_drop, se = paired(ds, [intact[k[2]] for k in keys])
            name = f"{cond[0]}:{cond[1]:g}"
            drops.setdefault(name, {})[model] = {k[2]: source[k] - intact[k[2]] for k in keys}
            t = mean_drop / se if se and se == se else float("nan")
            rows.append((mean_drop, f"{name + ('  (thesis)' if thesis else ''):26s} {len(keys):3d} "
                                    f"{st.mean(ds):6.2f} {mean_drop:+7.2f} {se:5.2f} {t:6.2f}"))
    for _, line in sorted(rows):
        print(line)
    print()

print("=== side by side: drop from each model's own intact score")
print(f"{'condition':22s} {'seed 2':>8s} {'reference':>10s} {'ref - seed2':>12s} {'SE':>5s}  n")
a, b = "dense_consistency_seed2", "reference_v150"
for name in sorted(drops, key=lambda n: drops[n].get(a) and st.mean(drops[n][a].values()) or 0):
    if a not in drops[name] or b not in drops[name]:
        continue
    common = sorted(set(drops[name][a]) & set(drops[name][b]))
    if len(common) < 2:
        continue
    da = [drops[name][a][r] for r in common]
    db = [drops[name][b][r] for r in common]
    diff, se = paired(db, da)
    print(f"{name:22s} {st.mean(da):+8.2f} {st.mean(db):+10.2f} {diff:+12.2f} {se:5.2f}  {len(common)}")
