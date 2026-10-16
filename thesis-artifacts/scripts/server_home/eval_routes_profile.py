import collections, math, pathlib
from xml.etree import ElementTree as ET
L = pathlib.Path.home() / "LEAD/lead"
bench = {p.name: p for p in (L / "src/lead/routes/benchmark_routes/bench2drive").rglob("*.xml")}
def prof(p):
    r = ET.parse(p).getroot(); route = r.find(".//route")
    wps = [(float(w.get("x")), float(w.get("y"))) for w in route.iter("position")]
    length = sum(math.dist(a, b) for a, b in zip(wps, wps[1:]))
    return route.get("town"), [s.get("type") for s in r.iter("scenario")], length
def summary(names, label):
    towns, types, lens = collections.Counter(), collections.Counter(), []
    for n in names:
        t, ts, l = prof(bench[n]); towns[t] += 1; lens.append(l)
        for x in set(ts): types[x] += 1
    lens.sort()
    print(f"== {label}: {len(names)} routes | towns {len(towns)} | scenario types {len(types)} | length m min/median/max {lens[0]:.0f}/{lens[len(lens)//2]:.0f}/{lens[-1]:.0f}")
    print("  towns:", dict(sorted(towns.items())))
    return types
names = (L / "src/lead/routes/eval_sets/degradation_30.txt").read_text().split()
t30 = summary(names, "degradation_30")
print("  types:", dict(sorted(t30.items())))
tall = summary(list(bench), "all bench2drive")
print("  types missing from the 30:", sorted(set(tall) - set(t30)))
