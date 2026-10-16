"""Mean driving score per model and condition, with the censored rows counted.

A route killed by the harness timeout carries no score and is not missing at
random -- a route is killed for taking too long, which is what a bad route
does -- so the count is printed beside every mean rather than folded into it.
"""

import collections
import csv
import pathlib
import statistics
import sys

for path in sys.argv[1:]:
    rows = list(csv.DictReader(pathlib.Path(path).open(encoding="utf-8")))
    print(f"\n=== {path} ({len(rows)} rows) ===")
    groups = collections.defaultdict(list)
    censored = collections.Counter()
    for row in rows:
        key = (row["model"], row["modality"], row["severity"])
        raw = (row.get("driving_score") or "").strip()
        try:
            groups[key].append(float(raw))
        except ValueError:
            censored[key] += 1
    for key in sorted(groups | censored):
        scored = groups.get(key, [])
        mean = f"{statistics.fmean(scored):6.2f}" if scored else "   n/a"
        model, modality, severity = key
        print(f"  {model:<14} {modality:<7} sev={severity:<4} "
              f"n={len(scored):>2} mean={mean}  censored={censored[key]}")
