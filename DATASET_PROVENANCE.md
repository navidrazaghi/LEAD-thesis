# Which data this project used, and how to get it again

The dataset is not in this backup. It is 25 GB and re-downloadable, and the
script that fetched it is in the repository. What is not re-derivable is
*which* part of it was used, so that is what this file records.

Written 2026-10-05, while the server that held it was still up.

## The source

| | |
| :--- | :--- |
| Hugging Face dataset | `ln2697/lead-123d` |
| Revision as used | `36d36c020a8838c531999105e65d8a84a33c676f` |
| Last modified | 2026-08-09 |
| Public | yes, no token needed |
| Fetcher | `scripts/common/fetch_dataset_subset.py`, in the repo |

**Pin the revision.** The fetcher resolves against `.../resolve/main`, so it
follows the branch. Nothing guarantees `main` still points at the revision above.
Re-fetching without pinning may give different files, and the log lists below
are the check on that: if a name in `training_logs_450.txt` no longer resolves,
the dataset moved and every number in the thesis is against a different subset.

## The two sets were fetched differently

This is the part that is easy to get wrong on a re-download.

| | logs | files each | cameras |
| :--- | ---: | ---: | :--- |
| Training subset | 450 | 19 | `f0`, `l0`, `r0` only |
| Held-out subset | 28 | 31 | all six |

The training fetch drops what the model never opens: LEAD reads three of the six
cameras, so the other three are pure download. The held-out fetch did not apply
that filter, which is why its logs are larger. Reproducing the training set
means reproducing the filter, not just the names.

A training log contains exactly:

```
box_detections_se3.arrow      camera.pcam_f0.arrow          ego_state_se3.arrow
camera_depth.pcam_f0.arrow    camera.pcam_l0.arrow          lidar.lidar_top.arrow
camera_depth.pcam_l0.arrow    camera.pcam_r0.arrow          radar.radar_merged.arrow
camera_depth.pcam_r0.arrow    camera_semantic.pcam_f0.arrow sync.arrow
camera_instance.pcam_f0.arrow camera_semantic.pcam_l0.arrow traffic_light_detections.arrow
camera_instance.pcam_l0.arrow camera_semantic.pcam_r0.arrow
camera_instance.pcam_r0.arrow custom.driving_meta.arrow
```

## The lists

| File | What it is |
| :--- | :--- |
| `training_logs_450.txt` | The 450 logs every rung trained on |
| `heldout_logs_28.txt` | The 28 the observability head was evaluated on |
| `all_logs_with_scenario.txt` | All 478 as `scenario/log`, giving each log its scenario type |
| `degradation_30.txt` | The 30 routes every closed-loop number was measured on |
| `eval_sets_available.txt` | The other route sets that existed |

**Keep the two sets apart.** `training.data.py123d_log_names` defaults to
*every log on disk*. The held-out logs live in the same tree, so any training
run that does not name the 450 explicitly will train on the held-out set too and
silently invalidate the observability result. That happened once here and was
caught only because the cache had no entry for them; with a cache present it
would have passed unnoticed. `run_post31.sh` and `run_rung2c.sh` both pin the
list and refuse to start if the count is not 450.

## Rebuilding

```bash
# 1. the subset, pinned. Edit REPO/resolve/main in the fetcher to the revision
#    above before running it.
python scripts/common/fetch_dataset_subset.py --budget 450 --jobs 4

# 2. check it against the record
find data/lead/123D/logs/normal_view -mindepth 2 -maxdepth 2 -type d -printf '%f\n' \
  | sort | diff - training_logs_450.txt

# 3. the lmdb cache the trainer reads. 1.7 GB, and hours of CPU, which is why
#    the log list matters more than the logs.
python -m lead.training.build_cache
```

The selection is stratified rather than random -- every adverse-weather route,
every route of a rare scenario type, then a round-robin over scenario types.
`scripts/common/select_training_subset.py` explains why a random slice is the
wrong subset. Given the same revision and budget it should reproduce the same
450, but the list here is the authority, not the algorithm.
