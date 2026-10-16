"""Which log view makes py123d skip route_position.arrow?"""
import logging, pathlib
from lead.log_reader import scene_index as si

hits = []
class Catch(logging.Handler):
    def emit(self, record):
        if "route_position" in record.getMessage():
            hits.append(record.getMessage())
logging.getLogger().addHandler(Catch())

ROOT = pathlib.Path.home() / "LEAD/lead/data/lead/123D/logs"
new = "Town01_Rep0_route_001547_route0_08_02_02_32_23"
cache = sorted(p.name for p in (ROOT.parent / "transfuser_training_cache/normal_view").iterdir())
old = next(n for n in cache if (ROOT / "perturbated_view").glob(f"*/{n}") and any((ROOT / "perturbated_view").glob(f"*/{n}")))
pairing = si.get_perturbated_view_pairing(ROOT.parent, [new, old])
for label, view, name in (("old log, normal", "normal_view", old), ("new log, normal", "normal_view", new),
                          ("old log, perturbated", "perturbated_view", old), ("new log, perturbated", "perturbated_view", new)):
    before = len(hits)
    d = next((ROOT / view).glob(f"*/{name}"))
    idx = si.build_log_scene_index(d, pairing._scene_filter)
    scene = si.ArrowSceneAPI(log_dir=idx.log_dir, scene_metadata=idx.scene_metadata_at(int(idx.anchor_indices[0])))
    _ = scene.available_camera_names
    print(f"{label:22s} warnings {len(hits) - before}  files: {'route_position.arrow' in {p.name for p in d.iterdir()}}")
print("sample:", hits[:1])
