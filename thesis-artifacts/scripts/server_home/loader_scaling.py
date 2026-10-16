# CPU-only: how fast can the data loader alone produce training batches, and
# what does each extra worker cost in RAM? No model forward, no GPU, no download.
import os, pathlib, sys, time, traceback
import torch
from torch.utils.data import DataLoader
from lead.config import LeadConfig
from lead.policy.transfuser.transfuser import Transfuser

def mem_avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1e6

cache = pathlib.Path('data/lead/123D/transfuser_training_cache/normal_view')
names = sorted(p.name for p in cache.iterdir())
assert len(names) == 450, len(names)
c = LeadConfig(); t = c.policy.transfuser
t.backbone_target = 'lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone'
t.deformable_calibrated_reference = True
t.use_planning_decoder = True
c.training.data.use_sensor_degradation = True
c.training.data.read_from_cache_store = True
c.training.data.py123d_log_names = names
try:
    ds = Transfuser(c).build_dataset()
except Exception:
    traceback.print_exc(); sys.exit(1)
print(f'dataset: {len(ds)} samples | RAM available at start: {mem_avail_gb():.1f} GB', flush=True)
print('workers | samples/s (loader alone) | startup s | RAM drop GB', flush=True)
for nw in (8, 16, 24, 30):
    try:
        dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=nw,
                        collate_fn=getattr(ds, 'collate_fn', None), pin_memory=False,
                        prefetch_factor=2, persistent_workers=False)
        before = mem_avail_gb(); low = before
        it = iter(dl); t0 = time.time()
        for _ in range(12): next(it)
        t1 = time.time()
        for i in range(120):
            next(it)
            if i % 10 == 0: low = min(low, mem_avail_gb())
        t2 = time.time()
        print(f'{nw:7d} | {120*32/(t2-t1):24.1f} | {t1-t0:9.1f} | {before-low:6.1f}', flush=True)
        del it, dl; time.sleep(5)
    except Exception:
        traceback.print_exc(); print(f'{nw} workers FAILED', flush=True)
print('DONE', flush=True)
