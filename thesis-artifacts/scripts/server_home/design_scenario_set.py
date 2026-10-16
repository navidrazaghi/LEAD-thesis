"""Second eval set: one route per scenario type, towns balanced, scored intact.

Clear weather alone cannot supply it: 87 of 220 routes are clear, the eval sets
already hold 60 of those, and the 27 left are all Town12 and cover 18 types.
So routes are drawn by preference -- free clear, then free adverse, then the
weather set as a last resort -- and each pick's weather is recorded. Never from
the 30 scored routes, nor from calibration (the governor's disjointness).
Uses select_eval_routes' own reader and adverse rule. Deterministic.
"""
import collections, pathlib, sys
from xml.etree import ElementTree as ET
L = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(L / "scripts/common"))
import select_eval_routes as S

sets = {f.stem: set(f.read_text().split()) for f in (L / "src/lead/routes/eval_sets").glob("*.txt")}
used = set().union(*sets.values())
routes = S.read_routes(L / "src/lead/routes/benchmark_routes/bench2drive")
typ = {r.path.name: {s.get("type") for s in ET.parse(r.path).getroot().iter("scenario")} for r in routes}
all_types = sorted(set().union(*typ.values()))

def tier(r):
    n = r.path.name
    if n in sets["degradation_30"] or n in sets["calibration"]:
        return None
    if n in sets["weather"]:
        return 2
    if n in used:
        return None          # degradation / pilot lists: leave them out too
    return 1 if r.is_adverse else 0

cands = collections.defaultdict(list)
for r in routes:
    k = tier(r)
    if k is not None:
        for t in typ[r.path.name]:
            cands[t].append((k, r))

per_town, taken, picked = collections.Counter(), set(), []
for t in sorted(all_types, key=lambda t: (len(cands[t]), t)):
    opts = [(k, r) for k, r in cands[t] if r.path.name not in taken]
    if not opts:
        continue
    k, r = min(opts, key=lambda kr: (kr[0], per_town[kr[1].town], kr[1].path.name))
    per_town[r.town] += 1; taken.add(r.path.name); picked.append((t, k, r))

label = {0: "clear", 1: "adverse", 2: "adverse (weather set)"}
print(f"picked {len(picked)}/{len(all_types)} types | towns {len(per_town)}: {dict(sorted(per_town.items()))}")
print("weather:", dict(collections.Counter(label[k] for _, k, _ in picked)))
print("types not covered:", [t for t in all_types if t not in {t for t, _, _ in picked}])
for t, k, r in sorted(picked):
    w = r.weather
    print(f"  {t:42s} {r.path.name:10s} {r.town:9s} {label[k]:22s} rain {w['rain']:.0f} fog {w['fog']:.0f} sun {w['sun']:.0f}")
out = pathlib.Path.home() / "new_subset/scenario_44.txt"
out.write_text("\n".join(sorted(r.path.name for _, _, r in picked)) + "\n")
print("written", out)
