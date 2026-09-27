# The replacement server, from 2026-09

The lab server that produced the thesis is no longer available, and its
trained weights went with it. This records how the
replacement was brought to the same state, what was checked to show it is the
same state, and the first results it produced. The scripts named here are in
`scripts/new_server_setup/` (building the machine) and `scripts/server_home/`
(everything run on it); the logs they wrote are in `logs/new_server/`.

## The machine

`ssh razaghi@172.20.27.6`, an internal Sharif address: reaching it needs a
connection to the campus network, and jobs keep running when that connection
drops. Ubuntu 24.04, one A100-SXM4-40GB, 16 cores, 62 GB RAM. Everything lives
on the large disk under `/home/new_drive/razaghi`. The home-directory paths
the campaign scripts hard-code -- `~/LEAD/lead`, `~/miniconda3`, `~/CARLA` --
are symlinks to it, so those scripts run unchanged.

## What differs from the old machine, and what that cost

- **Two system libraries were missing.** Without `ffmpeg` the evaluation agent
  refuses to set up, and `run_evaluation.py` records that as a real DS 0
  ("Agent couldn't be set up" is not in its infrastructure-failure list), so a
  whole sweep would have scored zero. `pyturbojpeg==2.5` needs libjpeg-turbo 3.x
  and Ubuntu 24.04 ships 2.1.5; the 3.2.0 library from conda-forge sits alone in
  `lib/turbojpeg3` and every run puts it first on `LD_LIBRARY_PATH`. Its decoding
  of dataset frames was checked to be bit-identical to other decoders.
- **The network.** github.com is unreachable from the server (the one git
  dependency, `py123d`, is served from a local mirror via `git insteadOf`); uv's
  default download concurrency fails on this link and 4 connections succeed;
  the captive portal drops for minutes to hours, so every downloader resumes.
- **The dataset branch moved.** See `DATASET_PROVENANCE.md`; everything was
  fetched at the pinned revision `36d36c0`.

## The data, verified

585 logs of `provenance/new_subset/selected_frames_town.txt`, both views: 19 files per normal-view log and 15
per perturbated-view log, 32.3 GB. The 585 names are identical, name for name,
to those pinned in `configs/rung0_diverse.yaml`. The training set comes to
60,703 samples, i.e. 948 batches of 64 = 60,672 per epoch, the campaign's
figure. Training logs report `Scene index: 585 logs in perturbated_view`.

## Checks that the pipeline behaves as it did

**The published checkpoint on our 30 routes, old server against new**
(`results/closed_loop_reference_v150.csv` against `reference_closed_loop.csv`):

| Condition | pairs | old | new | new - old | SE | identical routes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| intact | 30 | 90.71 | 94.82 | +4.11 | 4.23 | 24/30 |
| LiDAR destroyed | 27 | 82.40 | 86.25 | +3.85 | 4.71 | 17/27 |
| camera destroyed | 25 | 63.70 | 55.48 | -8.23 | 5.50 | 6/25 |

**Camera destroyed, the same checkpoint driven twice on the new server**
(`results/closed_loop_reference_v150_camera_rep2.csv`): run 1 against run 2,
+5.76 (SE 3.76), 8/29 routes identical, 12.46 points mean absolute difference
per route -- as large as old against new (18.00 and 13.66). The camera gap above
is run-to-run noise, not a machine difference. It is also a finding in its own
right: under camera destruction one model's closed-loop score moves by about 12
points per route from one run to the next, so single evaluations in that
condition carry a large uncertainty.

**Our own checkpoints load and drive** (`results/system_test_smoke_load.csv`):
a 60-batch post-train from a 60-batch pretrain drove 26.6 m and 25.9 m on two
routes. A pretrain-only checkpoint crashes the agent, as expected: it has no
planning decoder.

**The unit suite**: 1053 passed, 1 skipped, 1 failed (the runtime type-checking flag test,
which depends on an environment variable).

## Speed

Measured on this machine with the campaign recipe (`logs/new_server/speed_test.log`):
pretrain 81.8 samples/s, post-train 77.0 (old machine: 87 and 85). The GPU is
the limit (GPU 80-87% busy, CPU 73-75% idle). `torch_compile_mode=max-autotune`, which
LEAD's docs/speed.md recommends, gains 3.5-4.4% in throughput but spends about
21 minutes before its first step and peaks at 38.6 GB of GPU memory in post-train: it does
not pay for itself here.

## In flight when this was written (2026-09-27)

- Seed 2 of dense + curriculum v2 + consistency
  (`scripts/server_home/run_diverse_dense_consistency_seed2.sh`), plus a second
  camera-destroyed evaluation of it.
- The structured fault catalogue: 8 camera and 5 LiDAR modes after VG-SAF's
  catalogue (arXiv 2608.24366), evaluation-side, on branch
  `stage0-fault-catalog`. Every pre-existing corruption path was checked to
  produce byte-identical output with it in place
  (`logs/new_server/identity_before_merge.txt`); `figures/fault_catalog_*.png`
  show each mode on a real frame.
- Stage 0: the seed-2 model and the published checkpoint under all 13 faults
  at two severities on 10 routes (`scripts/server_home/stage0_run_v2.sh`).
- The observability gate trained as model 3 of the diverse campaign plus the
  gate, then scored with the learned gate and with the oracle gate on the same
  checkpoint (`scripts/server_home/run_diverse_gate.sh`).
