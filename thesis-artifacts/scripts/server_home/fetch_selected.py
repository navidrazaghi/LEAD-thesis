# -*- coding: utf-8 -*-
"""Download a designed selection: normal + perturbated view, the policy's cameras only.

Paths come from the full paginated listing (~/full_listing_sizes.json), so nothing
is re-listed. Transfers go through the repository's own download() (curl, resume,
skip-existing, retries).

Usage: python fetch_selected.py <selection.txt> [first N logs] [jobs]
"""

import collections
import json
import pathlib
import sys

HOME = pathlib.Path.home()
LEAD = HOME / "LEAD/lead"
sys.path.insert(0, str(LEAD / "scripts/common"))
from fetch_dataset_subset import DEFAULT_ENDPOINT, download  # noqa: E402

ROOT = LEAD / "data/lead/123D"
USED_CAMERAS = ("pcam_l0", "pcam_f0", "pcam_r0")
CAMERA_STREAMS = ("camera.", "camera_depth.", "camera_instance.", "camera_semantic.")


def wanted(filename: str) -> bool:
    if filename.startswith(CAMERA_STREAMS):
        return any(f".{camera}." in filename for camera in USED_CAMERAS)
    return True


selection = pathlib.Path(sys.argv[1]).read_text().split()
limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(selection)
jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
selection = selection[:limit]
keys = set(selection)

files = collections.defaultdict(list)
sizes = {}
for path, size in json.loads((HOME / "full_listing_sizes.json").read_text()):
    parts = path.split("/")
    if len(parts) >= 5 and parts[0] == "logs" and parts[1] in ("normal_view", "perturbated_view"):
        if f"{parts[2]}/{parts[3]}" in keys and wanted(parts[4]):
            files[parts[1]].append(path)
            sizes[path] = size

paths = files["normal_view"] + files["perturbated_view"]
missing = [p for p in paths if not (ROOT / p).exists()]
print(f"{len(selection)} logs: {len(files['normal_view'])} normal + {len(files['perturbated_view'])} "
      f"perturbated files, {sum(sizes[p] for p in missing)/1e9:.2f} GB still to fetch", flush=True)
failed = download(paths, ROOT, jobs, DEFAULT_ENDPOINT)
print(f"done, {failed} failed", flush=True)
sys.exit(1 if failed else 0)
