"""Does the 450-log cache already carry observability targets?

The ladder doc says use_observability decides whether those targets reach the
store, so a run that needs them against a store built without them would
rebuild the cache -- 450 logs, on a disk with 14 GB free. Checking is cheaper
than finding out.
"""

import pathlib
import lmdb

CACHE = pathlib.Path.home() / "LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view"

logs = sorted(p for p in CACHE.iterdir() if p.is_dir())
print(f"{len(logs)} cached logs")

for log in logs[:3]:
    env = lmdb.open(str(log), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        keys = [k.decode("utf-8", "replace") for k, _ in txn.cursor()]
    env.close()
    obs = sorted({k for k in keys if "observability" in k})
    # Keys are per-sample, so show the distinct suffixes rather than all of them.
    suffixes = sorted({k.split("/")[-1] if "/" in k else k for k in keys})
    print(f"\n{log.name}: {len(keys)} keys")
    print(f"  observability keys: {len(obs)}")
    if obs:
        print(f"    e.g. {obs[0]}")
    print(f"  distinct key shapes (first 25): {suffixes[:25]}")
