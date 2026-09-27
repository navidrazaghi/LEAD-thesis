"""How fast the data loader alone produces training batches on this machine.

Training on the old server was GPU-bound: its loader gave 556 samples/s with 16
workers against about 87 consumed by training. This machine has 16 cores to the
old one's 32, so that margin has to be measured again rather than assumed.
Same configuration as the campaign's dense pretrain: the 585 pinned logs, both
views, read from the cache store. No model, no GPU.

Adapted from thesis-artifacts/scripts/server_home/loader_scaling.py, which is
tied to the old 450-log cache.
"""
import pathlib
import time

from torch.utils.data import DataLoader

from lead.config import LeadConfig
from lead.policy.transfuser.transfuser import Transfuser

SEL = pathlib.Path("thesis-artifacts/provenance/new_subset/selected_frames_town.txt")
names = sorted(line.split("/")[1] for line in SEL.read_text().split() if line.strip())
assert len(names) == 585, len(names)

config = LeadConfig()
config.training.data.read_from_cache_store = True
config.training.data.py123d_log_names = names
dataset = Transfuser(config).build_dataset()
print(f"dataset: {len(dataset)} samples", flush=True)
print("workers | samples/s, loader alone | start-up s", flush=True)

for workers in (8, 16):
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True,
                        num_workers=workers, collate_fn=getattr(dataset, "collate_fn", None),
                        pin_memory=False, prefetch_factor=2, persistent_workers=False)
    batches = iter(loader)
    t0 = time.time()
    for _ in range(12):
        next(batches)
    t1 = time.time()
    for _ in range(120):
        next(batches)
    t2 = time.time()
    print(f"{workers:7d} | {120 * 32 / (t2 - t1):24.1f} | {t1 - t0:10.1f}", flush=True)
    del batches, loader
    time.sleep(5)
print("DONE", flush=True)
