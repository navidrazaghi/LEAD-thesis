# Seed replicates of dense + curriculum v2 + consistency

The lab server was lost on 2026-10-17 while the seed-2 replicate was training.
The seed-1 run had finished; its results CSV never left the machine, so the
numbers below are the record of it. They were computed on the server from
`results/closed_loop_diverse_dense_consistency_seed1.csv` with the same paired
procedure as every other comparison in this repository: per-route difference on
routes both models scored, mean, standard error, `t`, and win/tie/loss.

Everything here is a transcription of that output, not a re-derivation. The
seed-0 run's own CSV is in this directory
(`closed_loop_diverse_dense_consistency.csv`) and is unaffected.

## What each run was

| | seed 0 | seed 1 | seed 2 |
| :--- | :--- | :--- | :--- |
| Pretrain | shared with the dense + curriculum v2 run | its own, `training.experiment.seed=1` | its own, `training.experiment.seed=2` |
| Post-train | 31 epochs, consistency 0.1 | 31 epochs, consistency 0.1 | 31 epochs, consistency 0.1 (replacement server) |
| Closed loop | 30 routes x 3 conditions | 30 routes x 3 conditions | 30 routes x 3 conditions, camera twice |

Everything else was identical: the 585-log subset, curriculum v2 flags, 31+31
epochs, batch 32 x accumulation 2, and `evaluation.inference.degrade_seed=0`, so
both seeds were scored under the same damage stream.

Launchers: `../scripts/server_home/run_diverse_dense_consistency.sh` (seed 0) and
`run_diverse_dense_consistency_seed1.sh` (seed 1). The seed-2 launcher was that
file with `SEED=1` -> `SEED=2` and `seed1` -> `seed2` throughout; it is not kept
here because it produced no result.

## Seed 1, closed loop

Mean driving score, with the routes that scored in brackets:

| Condition | seed 0 | seed 1 |
| :--- | ---: | ---: |
| Intact | 55.75 (30) | 71.96 (29) |
| LiDAR destroyed | 60.38 (30) | 69.59 (30) |
| Camera destroyed | 57.62 (30) | 43.24 (30) |

Route completion and infraction penalty, seed 1: intact 88.1% / 0.811, LiDAR
destroyed 86.5% / 0.812, camera destroyed 71.8% / 0.584. Routes completed: 21,
21 and 10 of 30. Run outcome: 89 of 90 scored, one `NoResult` (intact, route
11715), 10 `TickRuntime`, 21 route deviations, 6 blocked.

## Paired differences

Seed 1 minus seed 0, the same model trained twice:

| Condition | mean | SE | t | W/T/L | n |
| :--- | ---: | ---: | ---: | :--- | ---: |
| Intact | +14.48 | 6.53 | 2.22 | 12/7/10 | 29 |
| LiDAR destroyed | +9.21 | 6.95 | 1.33 | 13/9/8 | 30 |
| Camera destroyed | -14.38 | 7.32 | -1.97 | 9/3/18 | 30 |

Seed 1 against the two models it is normally read against:

| Comparison | Intact | LiDAR destroyed | Camera destroyed |
| :--- | ---: | ---: | ---: |
| vs dense + curriculum v2 (seed 0 pretrain) | +13.36 (SE 4.89, t 2.73) | +0.04 (SE 5.78) | -1.51 (SE 5.02) |
| vs baseline | +3.99 (SE 6.21) | +19.42 (SE 5.84, t 3.33) | -3.99 (SE 7.28) |

Fall from each model's own intact score: seed 0 gave -4.63 (LiDAR) and -1.87
(camera); seed 1 gave +2.38 and +28.57.

## What this measures, and what it does not

Training is stable across seeds: the final training objective was 0.1808 /
0.1668 (seed 0 pretrain / post-train) against 0.1796 / 0.1758 for seed 1, and the
per-epoch curves are in `training_curves_consistency_seeds.csv`. Closed-loop
driving is not: the same recipe moved 14 to 16 points between seeds, which is
larger than most effects reported in this work and larger than the standard
errors of the paired route comparisons, which capture route variability only.

Two consequences for how the results in this repository should be read:

1. Differences of the order of 20 points or more, such as the deformable
   operator's LiDAR-off gain (+26.05) and the curriculum's (+20.05), stand well
   clear of this spread.
2. The consistency term's camera-off gain at seed 0 (+13.14, p = 0.06) does not.
   At seed 1 the same comparison is -1.51. It should be reported as within seed
   variability, not as an effect of the term.

This experiment does not separate training variability from the simulator's own
run-to-run variability; that would need one model evaluated twice, which was
queued and never ran.

## Timing, for planning a rerun

Seed 1 on one A100: pretrain 5 h 57 min, post-train with the consistency term
6 h 26 min, parallel closed-loop evaluation 2 h 51 min, 15 h 14 min in total. The
consistency term cost about 13% in training throughput (76 against 85 samples/s).

## Queued and never run

`../scripts/server_home/run_after_seed2.sh` holds the two experiments that were
waiting behind seed 2, with every parameter as it would have run:

- held-out loss of the curriculum models, to see whether the degradation
  curriculum narrows the 4.7x generalisation gap the baseline has;
- an 850-log subset at the same 60,672 scenes per epoch, to move log diversity
  with compute held fixed.

## Seed 2, on the replacement server (2026-09-28)

Trained and scored on the replacement machine (`../NEW_SERVER.md`) with the same
585 logs at the pinned dataset revision, the same recipe and the same 30 routes.
Results: `closed_loop_diverse_dense_consistency_seed2.csv`, and a second
camera-destroyed evaluation of the same checkpoint in
`closed_loop_diverse_dense_consistency_seed2_camera_rep2.csv`. Final training
objective 0.1793 (pretrain) and 0.1636 (post-train). Every number below is
printed by `../scripts/seed_three_analysis.py`, which first reproduces the
two-seed 16.2 / 9.1 / 20.7 as a check that it uses the thesis's method.

| Condition | seed 0 | seed 1 | seed 2 | mean | sd | baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intact | 55.75 | 71.96 | 68.08 | 65.26 | 8.46 | 66.44 |
| LiDAR destroyed | 60.38 | 69.59 | 73.14 | 67.70 | 6.59 | 50.17 |
| Camera destroyed | 57.62 | 43.24 | 41.94 | 47.60 | 8.70 | 45.27 |

- Seed 2 minus seed 0, paired: +12.32 (SE 6.60), +12.12 (SE 7.47), -15.26 (SE 7.92).
  Seeds 1 and 2 agree; seed 0 is the outlier in all three conditions.
- LiDAR destroyed against the baseline: seed 0 +10.21 (t 1.33), seed 1 +19.42
  (t 3.33), seed 2 +22.03 (t 4.03). The camera-destroyed gain of seed 0 did not
  replicate (3-seed mean +2.33 over the baseline).
- The consistency term against dense + curriculum v2, camera destroyed: +13.14
  (seed 0), -1.51 (seed 1), -2.32 (seed 2, SE 6.89).
- The same checkpoint driven twice, camera destroyed: the reference +5.76 (SE
  3.76, 12.46 per route, 8/29 identical), seed 2 +1.47 (SE 4.24, 17.25 per
  route, 3/29 identical). Implied sd of one evaluation's mean: 2.97, about 14% of
  the seed-to-seed variance.
- Seed-aware minimum detectable difference: per-run sd 8.0, threshold 19.7
  (two-seed estimate 9.1 and 20.7). Three seeds is itself a noisy estimate.
