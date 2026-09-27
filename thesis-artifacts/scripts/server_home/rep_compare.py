"""Camera-destroyed, three runs of one checkpoint: old server, new run 1, new run 2.

The question is whether the old-vs-new gap in this condition is larger than the
gap between two runs on the same machine. Each pair is compared on the routes
both runs scored; a route with no score in either is named, not dropped.
"""
import csv
import pathlib
import statistics as st

R = pathlib.Path("/home/new_drive/razaghi/lead")
COND = ("camera", 1.0)


def load(path, model):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["model"] == model and (r["modality"], float(r["severity"] or 0)) == COND:
            out[r["route"]] = r
    return out


runs = {
    "old": load(R / "thesis-artifacts/results/reference_closed_loop.csv", "ref0"),
    "new1": load(R / "results/closed_loop_reference_v150.csv", "reference_v150"),
    "new2": load(R / "results/closed_loop_reference_v150_camera_rep2.csv", "reference_v150_rep2"),
}
for name, rows in runs.items():
    scored = [float(r["driving_score"]) for r in rows.values() if r["driving_score"]]
    print(f"{name:5s} {len(rows)} rows, {len(scored)} scored, mean DS {st.mean(scored):6.2f}")

print()
for a, b in (("new1", "new2"), ("old", "new1"), ("old", "new2")):
    routes = sorted(k for k in runs[a] if k in runs[b]
                    and runs[a][k]["driving_score"] and runs[b][k]["driving_score"])
    d = [float(runs[b][k]["driving_score"]) - float(runs[a][k]["driving_score"]) for k in routes]
    se = st.stdev(d) / len(d) ** 0.5
    same = sum(abs(x) < 1e-9 for x in d)
    mad = st.mean(abs(x) for x in d)
    print(f"{a:5s} vs {b:5s}  n={len(d):2d}  mean diff {st.mean(d):+6.2f}  SE {se:5.2f}  "
          f"t {st.mean(d) / se:5.2f}  identical {same:2d}/{len(d)}  mean |diff| {mad:5.2f}")
print("\nRead: if 'new1 vs new2' looks like the other two rows, the old-vs-new gap is run-to-run "
      "noise; if it is much tighter, the machines differ in the camera path.")
