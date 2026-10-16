import csv, pathlib, statistics as st
from xml.etree import ElementTree as ET
L = pathlib.Path.home() / "LEAD/lead"
names = (L / "src/lead/routes/eval_sets/degradation_30.txt").read_text().split()
bench = {p.name: p for p in (L / "src/lead/routes/benchmark_routes/bench2drive").rglob("*.xml")}
def scores(f, cond="none"):
    d = {}
    for r in csv.DictReader(open(L / "results" / f)):
        if r["modality"] == cond and r["route"] in names:
            d[r["route"]] = float(r["driving_score"])
    return d
ref, ours = scores("reference_closed_loop.csv"), scores("closed_loop_post31.csv")
print(f"scored: reference {len(ref)}/30, post31 {len(ours)}/30")
rows = []
for n in names:
    root = ET.parse(bench[n]).getroot()
    typ = ",".join(sorted({s.get("type") for s in root.iter("scenario")}))
    rows.append((ref.get(n), ours.get(n), n, root.find(".//route").get("town"), typ))
rows.sort(key=lambda r: (r[0] is None, -(r[0] or 0)))
for rf, ou, n, tw, typ in rows:
    f = lambda v: "  -  " if v is None else f"{v:5.1f}"
    print(f"{f(rf)} {f(ou)}  {n:10s} {tw:9s} {typ}")
rv = [r[0] for r in rows if r[0] is not None]
print(f"reference: {sum(v >= 90 for v in rv)} routes >=90, {sum(50 <= v < 90 for v in rv)} in 50-90, {sum(v < 50 for v in rv)} <50; median {st.median(rv):.1f}")
ov = [r[1] for r in rows if r[1] is not None]
print(f"post31:    {sum(v >= 90 for v in ov)} routes >=90, {sum(50 <= v < 90 for v in ov)} in 50-90, {sum(v < 50 for v in ov)} <50; median {st.median(ov):.1f}")
