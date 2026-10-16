import collections, pathlib, sys
from xml.etree import ElementTree as ET
L = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(L / "scripts/common"))
import select_eval_routes as S
sets = {f.stem: set(f.read_text().split()) for f in (L / "src/lead/routes/eval_sets").glob("*.txt")}
print({k: len(v) for k, v in sets.items()})
routes = S.read_routes(L / "src/lead/routes/benchmark_routes/bench2drive")
cnt = collections.defaultdict(collections.Counter)
towns = collections.defaultdict(set)
for r in routes:
    n = r.path.name
    if n in sets["degradation_30"]:
        pool = "scored30"
    elif n in sets.get("calibration", ()):
        pool = "calib"
    elif n in sets.get("weather", ()):
        pool = "weather"
    else:
        pool = "free_adv" if r.is_adverse else "free_clear"
    for t in {s.get("type") for s in ET.parse(r.path).getroot().iter("scenario")}:
        cnt[t][pool] += 1
        if pool != "scored30":
            towns[t].add(r.town)
cols = ["free_clear", "free_adv", "calib", "weather", "scored30"]
print(f"{'type':44s} " + " ".join(f"{c:>10s}" for c in cols) + "  towns(non-scored)")
for t in sorted(cnt):
    print(f"{t:44s} " + " ".join(f"{cnt[t][c]:10d}" for c in cols) + "  " + ",".join(sorted(towns[t])))
print("types with no non-scored route at all:", [t for t in cnt if not towns[t]])
