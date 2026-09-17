# Picking this up on another machine

Written 2026-10-05, while the lab server that produced all of it was still up
and four days from being taken away. Everything here is what could not be
regenerated from the code alone.

## What is in this directory

| | |
| :--- | :--- |
| `results/` | Every number in the thesis. 45 files, the closed-loop sweeps among them |
| `configs/` | The resolved config of all 23 training runs, one per rung and stage |
| `provenance/` | Which data was used: see `DATASET_PROVENANCE.md` at the repo root |
| `logs/` | 56 driver and evaluation logs, for timings and for what actually happened |
| `scripts/` | The launchers, as they were run |

## What is not here, and why

**The trained weights.** 23 checkpoints, 5.5 GB, too large for a git repo. They
were on the server at `outputs/<run>/model_*.pth` and a partial copy went to a
laptop. Losing them costs 9 hours of training per rung to reproduce, not the
result itself: the configs here plus the code define every run exactly.

**The optimiser states**, 6.5 GB, which only matter for resuming a training run
that was interrupted. Starting a fine-tune fresh does not need them.

**The dataset**, 25 GB, re-downloadable. `DATASET_PROVENANCE.md` records the
Hugging Face repository, the exact revision, both log lists and the per-log file
filter, which is the part a naive re-download gets wrong.

**The raw W&B logs**, 2.2 GB of offline run files. What the thesis reads off
them is already extracted into `results/training_curves.csv`; the raw files
would only be needed to plot something nobody has plotted yet.

## Restoring enough to run

```bash
git clone <this repo> lead && cd lead
git checkout robust-deployment

# the dataset, pinned to the revision the thesis used
python scripts/common/fetch_dataset_subset.py --budget 450 --jobs 4
find data/lead/123D/logs/normal_view -mindepth 2 -maxdepth 2 -type d -printf '%f\n' \
  | sort | diff - thesis-artifacts/provenance/training_logs_450.txt

# the lmdb cache the trainer reads, hours of CPU
python -m lead.training.build_cache

# CARLA 0.9.16, 44 GB
bash thesis-artifacts/scripts/get_carla_0916.sh
```

Two environment variables every run needs, or it fails in ways that look
unrelated:

```bash
export TIMM_USE_OLD_CACHE=1              # the HF weights are in ~/.cache/torch
export LEAD_RUNTIME_TYPE_CHECKING=false  # beartype and Dynamo cannot co-exist
export WANDB_MODE=offline                # every run here logged offline
ulimit -n 65536                          # 8 loader workers exceed the 1024 default
```

That last one is not optional. Below it the first epoch dies on "Too many open
files" and the trainer then hangs at step 0 with the GPU idle, which reads like
a deadlock rather than a limit.

## The trap that cost the most

`training.data.py123d_log_names` defaults to **every log on disk**. The 28
held-out logs for the observability evaluation live in the same tree as the 450
training logs. Any training run that does not name the 450 explicitly trains on
the held-out set too, and the observability result silently stops measuring
generalisation.

It surfaced once here only because the cache had no entry for those logs and
lmdb raised. With a cache present it would have passed unnoticed.
`run_post31.sh` and `run_rung2c.sh` both pin the list and refuse to start if the
count is not 450. Copy that guard into anything new.

## Where the work stood

The thesis is complete: the seven-rung ladder in `results/closed_loop.csv`, the
reference comparison in `results/reference_closed_loop.csv`, the observability
evaluation in `results/observability_head_*.json`.

Three things were in flight when the server went away, and their result files
will be present here only if they finished in time:

- `closed_loop_post31.csv` — the retrained baseline with the published
  31-epoch post-train instead of the ladder's 10, testing whether the stall rate
  of 78 % against the reference checkpoint's 4 % comes from the halved recipe.
- `closed_loop_rung2c.csv` and `closed_loop_rung2c_substituted.csv` — rung 2a
  plus the cross-modal hallucination head, testing whether rebuilding a
  destroyed LiDAR grid from the camera helps where reweighting it did not.
- The oracle gate, `evaluation.inference.oracle_gate`, implemented and tested
  but never run: it hands the gate the reliability the harness applied, so a
  null result there would show the estimator was not what failed.

`docs/baseline_convergence.md` and the thesis text carry the reasoning behind
all three.

## Status on 2026-10-16: the diverse-subset campaign

Everything below was added after the section above was written. It is enough to
continue from where the server stopped, without the server.

### What ran

All on the 585-log diverse subset (`provenance/new_subset/selected_frames_town.txt`),
same recipe, 31 pretrain + 31 post-train epochs each, closed loop on the 30
degradation routes x {intact, LiDAR destroyed, camera destroyed}.

| Model | Output dirs (server) | Config | Closed-loop CSV |
| :--- | :--- | :--- | :--- |
| Baseline, dense fusion | `rung0_diverse`, `rung0_diverse_post31` | `configs/rung0_diverse*.yaml` | `results/closed_loop_diverse_baseline.csv` |
| Deformable fusion | `rung2ad_diverse`, `rung2ad_diverse_post31` | `configs/rung2ad_diverse*.yaml` | `results/closed_loop_diverse_deformable.csv` |
| Deformable + curriculum v2 | `rung2ad_diverse_curriculum2`, `..._post31` | `configs/rung2ad_diverse_curriculum2*.yaml` | `results/closed_loop_diverse_curriculum2.csv` |
| Dense + curriculum v2 | `rung2a_diverse_curriculum2`, `..._post31` | `configs/rung2a_diverse_curriculum2*.yaml` | `results/closed_loop_diverse_dense_curriculum2.csv` |
| Dense + curriculum v2 + consistency 0.1 | `rung2a_diverse_consistency_post31` (post-train from the dense + curriculum v2 pretrain) | `configs/rung2a_diverse_consistency_post31.yaml` | `results/closed_loop_diverse_dense_consistency.csv` |

Also: `results/closed_loop_scenario44.csv` (44-route one-per-scenario set,
baseline and reference), `results/heldout_loss.csv` (28 held-out logs),
`results/parallel_validation.csv` (parallel vs sequential evaluation agreement),
`results/training_curves_diverse_4models.csv` (per-epoch training objective).

### Code added in this campaign (this commit series)

- Curriculum v2 in `src/lead/policy/transfuser/utils/sensor_degradation.py`:
  independent per-modality draws, a full-failure probability, and camera/LiDAR
  misalignment. Flags in `src/lead/config/training/data_config.py`, all default
  off and draw-for-draw identical to the old curriculum when off.
- Degradation-consistency distillation: `training.data.degradation_consistency_weight`
  (default 0). `augment_batch_with_clean` in `transfuser.py` builds the damaged
  student batch and a clean teacher copy; `_degradation_consistency` in
  `train.py` adds an L1 term on waypoints and route, damaged samples only,
  teacher without gradient.
- `scripts/common/run_evaluation.py --work-dir`, so parallel shards do not share
  (and wipe) one scratch directory.
- `src/lead/routes/eval_sets/scenario_44.txt`, one Bench2Drive route per scenario type.

### Scripts as they were run

`scripts/server_home/` is every driver and analysis script from the server home
directory touched since 2026-10-05, unedited. The ones that matter to continue:

| Script | What it does |
| :--- | :--- |
| `design_subset.py`, `fetch_selected.py` | Build and download the diverse subset |
| `run_diverse_*.sh` | Train + evaluate one model of the table above |
| `eval_parallel.py` | Sharded closed-loop evaluation (3 CARLA instances, ports 400 apart) |
| `heldout_loss.py` | Loss on the 28 held-out logs |
| `live_loss_plot.py`, `curves_dump.py` | Training curves from the offline W&B logs |
| `modality_intervention.py` | Open-loop plan displacement when one sensor is destroyed |
| `verify_deformable.py`, `verify_curriculum_v2.py`, `verify_consistency.py` | Checks run before training |
| `model_view.py` | Export what the policy receives for one sample |

Chain logs of these runs are in `logs/diverse/`.

### Not in git

The checkpoints of the five models above (`outputs/<run>/model_0030.pth`), for
the same size reason as before. Without them every result above is reproducible
from the configs, at 11.5 hours (dense) to 16.7 hours (deformable) of training per model on one A100.

## After the server was lost (2026-10-17)

Access to the lab server ended while a seed-2 replicate was training. What that
leaves, and where it is:

| | |
| :--- | :--- |
| `results/seed_replicates_record.md` | The seed-1 replicate in full — its CSV never left the machine, so this is the record of it — and what the seed spread means for every other comparison here |
| `results/training_curves_consistency_seeds.csv` | Per-epoch training objective, seeds 0 and 1, both stages |
| `results/five_models_table.csv` | The five diverse-subset models side by side: score, route completion, infraction penalty, scored routes, drops |
| `figures/` | The three figures made from those tables |
| `scripts/server_home/run_after_seed2.sh` | The two experiments that were queued and never ran, with every parameter as it would have run |

Lost with the machine: the trained weights of all models, the seed-1 results
CSV (transcribed above), and the unfinished seed-2 run. Every result reported in
the thesis is in `results/`; the configs here define each run exactly, so any of
them can be reproduced on other hardware at 11.5 to 16.7 hours per model.
