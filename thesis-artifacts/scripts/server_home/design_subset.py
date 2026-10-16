# -*- coding: utf-8 -*-
"""Design a diverse training subset from the full LEAD 123D release.

Reads the paginated release listing with file sizes (~/full_listing_sizes.json)
and the local dataset, then writes a selection and a report. Downloads nothing.

Why a new subset. The 450 logs every run so far trained on came from a selector
with two defects: its repository listing was capped part way through the
alphabet, so it saw 14 of the release's 43 scenario types; and the town was a
sort key rather than a rotation axis, so two towns took 85% of the budget and
six of twelve contributed nothing. Against the scored routes, 20 of the 30
degradation routes test a scenario or town training never saw. And no log had
its perturbated view on disk, so the published recipe's 50% rig-perturbation
augmentation -- on in the reference checkpoint's own config.yaml -- never fired.

Candidates are complete normal-view logs whose perturbated view exists for the
three cameras the policy reads, minus the 28 logs held out for the
observability evaluation. Three cameras are enough: py123d's `camera:all`
requirement asks only that some camera column exist in the sync table, and the
normal view has trained on three of six cameras in every run so far.

Two orderings:
  type  -- round-robin over scenario types; within a type, the town with the
           fewest picks. Balances scenarios exactly, towns only as far as each
           type's town coverage allows.
  town  -- round-robin over towns, fewest picks first; within a town, the
           scenario type with the fewest picks overall. Balances towns as far
           as each town's supply allows, scenarios second.
Within a (type, town) cell: logs already on disk first -- no download -- then
length closest to the pool median, then name.

Two sizes:
  N       -- a fixed number of logs.
  frames  -- keep adding until the frame total reaches the current subset's
             312,502, so training compute per epoch matches the runs it will be
             compared against.

Weather is not a selection axis: collection runs with shuffle_weather, which
replaces the route's weather with a sampled preset recorded in the log. The
current 450 are 78% adverse, and any subset inherits that.

Usage: python design_subset.py <N | frames> <type | town>
"""

import collections
import json
import pathlib
import re
import statistics as st
import sys
from xml.etree import ElementTree as ET

HOME = pathlib.Path.home()
LEAD = HOME / "LEAD/lead"
DATA = LEAD / "data/lead/123D"
OUT = HOME / "new_subset"

LOG_NAME = re.compile(r"^(?P<town>Town\w+?)_Rep\d+_(?P<stem>.+)_route0_[\d_]+$")
REQUIRED = {"sync.arrow", "ego_state_se3.arrow", "lidar.lidar_top.arrow"}
USED_CAMERAS = ("pcam_l0", "pcam_f0", "pcam_r0")
CAMERA_STREAMS = ("camera.", "camera_depth.", "camera_instance.", "camera_semantic.")

SAMPLES_PER_FRAME = 58976 / 312502  # the trainer's samples over the current 450's frames
CURRENT_FRAMES = 312502
SAMPLES_PER_SECOND = 55.0           # measured recipe throughput, GPU-bound
EPOCHS = 31 + 31
DOWNLOAD_BYTES_PER_SECOND = 2.0e6   # measured on the held-out fetch

size_arg = sys.argv[1] if len(sys.argv) > 1 else "450"
MODE = sys.argv[2] if len(sys.argv) > 2 else "type"
assert MODE in ("type", "town"), MODE
MATCH_FRAMES = size_arg == "frames"
N_TARGET = 0 if MATCH_FRAMES else int(size_arg)
TAG = f"{size_arg}_{MODE}"


def frames_of(sync_bytes: int) -> float:
    """Fitted on the current 450: sync.arrow bytes -> frames, R^2 = 1.0000."""
    return max(0.0, 0.00375 * sync_bytes - 26.3)


def wanted(filename: str) -> bool:
    """fetch_dataset_subset.wanted_files' filter, depth kept."""
    if filename.startswith(CAMERA_STREAMS):
        return any(f".{camera}." in filename for camera in USED_CAMERAS)
    return True


# --- the release -------------------------------------------------------------
listing = json.loads((HOME / "full_listing_sizes.json").read_text())
views = {"normal_view": collections.defaultdict(dict), "perturbated_view": collections.defaultdict(dict)}
for path, size in listing:
    parts = path.split("/")
    if len(parts) >= 5 and parts[0] == "logs" and parts[1] in views:
        views[parts[1]][(parts[2], parts[3])][parts[4]] = size
normal, pert = views["normal_view"], views["perturbated_view"]

# --- what is already here ----------------------------------------------------
cache = {p.name for p in (DATA / "transfuser_training_cache/normal_view").iterdir()}
on_disk = {(d.parent.name, d.name) for d in (DATA / "logs/normal_view").glob("*/*") if d.is_dir()}
held_out = {k for k in on_disk if k[1] not in cache}
current = {k for k in on_disk if k[1] in cache}

pool = []
for key, files in normal.items():
    scenario, name = key
    match = LOG_NAME.match(name)
    if not match or not REQUIRED <= files.keys() or key in held_out:
        continue
    pfiles = pert.get(key, {})
    if not all(any(f.startswith("camera.") and f".{c}." in f for f in pfiles) for c in USED_CAMERAS):
        continue
    pool.append({
        "scenario": scenario, "name": name, "town": match.group("town"),
        "frames": frames_of(files["sync.arrow"]),
        "normal_bytes": sum(s for f, s in files.items() if wanted(f)),
        "pert_bytes": sum(s for f, s in pfiles.items() if wanted(f)),
        "pert_bytes_all": sum(pfiles.values()),
        "on_disk": key in on_disk, "current": key in current,
    })

median_frames = st.median(r["frames"] for r in pool)
cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
for r in pool:
    cells[(r["scenario"], r["town"])].append(r)
for k in cells:
    cells[k].sort(key=lambda r: (not r["on_disk"], abs(r["frames"] - median_frames), r["name"]))
types = sorted({r["scenario"] for r in pool})
towns = sorted({r["town"] for r in pool})

# --- selection -----------------------------------------------------------------
picked: list[dict] = []
per_cell: collections.Counter = collections.Counter()
per_town: collections.Counter = collections.Counter()
per_type: collections.Counter = collections.Counter()
frames_total = 0.0


def done() -> bool:
    return frames_total >= CURRENT_FRAMES if MATCH_FRAMES else len(picked) >= N_TARGET


def take(t: str, tw: str) -> None:
    global frames_total
    r = cells[(t, tw)].pop(0)
    picked.append(r)
    per_cell[(t, tw)] += 1
    per_town[tw] += 1
    per_type[t] += 1
    frames_total += r["frames"]


while not done():
    progressed = False
    if MODE == "town":
        for tw in sorted(towns, key=lambda x: (per_town[x], x)):
            if done():
                break
            available = [t for t in types if cells[(t, tw)]]
            if available:
                take(min(available, key=lambda x: (per_type[x], per_cell[(x, tw)], x)), tw)
                progressed = True
    else:
        for t in types:
            if done():
                break
            available = [tw for tw in towns if cells[(t, tw)]]
            if available:
                take(t, min(available, key=lambda x: (per_cell[(t, x)], per_town[x], x)))
                progressed = True
    if not progressed:
        break

# --- report ----------------------------------------------------------------------
samples = SAMPLES_PER_FRAME * frames_total
hours = samples * EPOCHS / SAMPLES_PER_SECOND / 3600
dl_normal = sum(r["normal_bytes"] for r in picked if not r["on_disk"])
dl_pert = sum(r["pert_bytes"] for r in picked)
reused = sum(r["on_disk"] for r in picked)
top2 = sum(n for _, n in per_town.most_common(2)) / len(picked)

print(f"=== [{TAG}] {len(picked)} logs ===")
print(f"  scenario types {len(per_type)}/43 (logs per type {min(per_type.values())}-{max(per_type.values())}) | "
      f"towns {len(per_town)}/12 | cells {len(per_cell)} | two largest towns {100*top2:.0f}%")
print("  per town: " + ", ".join(f"{t}={per_town[t]}" for t in towns))
print(f"  frames {frames_total:,.0f} (current {CURRENT_FRAMES:,}) -> {samples:,.0f} samples/epoch -> "
      f"{hours:.1f} h for 31+31 epochs")
print(f"  download {(dl_normal+dl_pert)/1e9:.1f} GB (normal {dl_normal/1e9:.1f} for {len(picked)-reused} new logs, "
      f"perturbated {dl_pert/1e9:.1f}) -> {(dl_normal+dl_pert)/DOWNLOAD_BYTES_PER_SECOND/3600:.1f} h at 2 MB/s | "
      f"reused from disk {reused}")

bench = {p.name: p for p in (LEAD / "src/lead/routes/benchmark_routes/bench2drive").rglob("*.xml")}
sel_types, sel_towns = set(per_type), set(per_town)
cur_types = {k[0] for k in current}
cur_towns = {LOG_NAME.match(k[1]).group("town") for k in current}
cov = []
for setname in ("degradation_30", "weather"):
    names = (LEAD / f"src/lead/routes/eval_sets/{setname}.txt").read_text().split()
    def seen(types_: set, towns_: set) -> int:
        n = 0
        for rn in names:
            root = ET.parse(bench[rn]).getroot()
            ts = [s.get("type") for s in root.iter("scenario")]
            n += root.find(".//route").get("town") in towns_ and any(t in types_ for t in ts)
        return n
    cov.append(f"{setname} {seen(cur_types, cur_towns)}->{seen(sel_types, sel_towns)}/{len(names)}")
print("  scored routes with both town and scenario seen (current->new): " + ", ".join(cov))

OUT.mkdir(exist_ok=True)
(OUT / f"selected_{TAG}.txt").write_text("\n".join(f"{r['scenario']}/{r['name']}" for r in picked) + "\n")
(OUT / f"selected_{TAG}.json").write_text(json.dumps(picked, indent=1))
if TAG == "450_type":
    print("  pool per town: " + ", ".join(f"{t}={sum(1 for r in pool if r['town']==t)}" for t in towns))
