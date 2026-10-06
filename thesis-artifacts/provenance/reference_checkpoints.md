# The reference checkpoints, and what they actually are

`reference/` holds three trained models that four result files are read
against. They are not in this repository — 790 MB — and until this file
existed, nothing here said what they were. Written 2026-10-06, while the
server that holds them was still up.

## They are the LEAD authors' checkpoints, not ours

Every one of the three records an `output_dir` under `/home/lnguyen/code/lead`,
which is the upstream author's machine, not this one. We did not train them; we
downloaded them and fetched them into `reference/` on 2026-09-25. They are the
published model this thesis's baseline is supposed to reproduce, which is the
whole reason the comparison exists.

All three are post-trained models with the planning decoder on, dense
`TransfuserBackbone`, 31 epochs at batch 64 with no accumulation — the
published recipe. That is worth stating because it is the recipe post31 was
built to match, and matching it was what took the corrected baseline from
18.77 to 43.20.

## The directory names do not mean what they look like

This is the trap, and it is the reason this file exists.

| directory         | `seed:` in its config | upstream run                                            |
| :---------------- | :-------------------- | :------------------------------------------------------ |
| `reference/seed0` | 0                     | `001_training/posttrain/260807_133603`                  |
| `reference/seed1` | **2**                 | `001_training/posttrain_seed2/260810_074247`            |
| `reference/seed2` | **0**                 | `007_fast_posttrain/posttrain_lru_cached/260808_192854` |

So `seed1` is seed 2, and `seed2` is not a third seed at all: it reports seed 0,
the same as `seed0`, and comes from a different experiment line — a fast
post-train with an LRU-cached loader rather than the plain one. Two of the three
are the same seed from different pipelines.

Anything that treats these as a clean three-seed spread is wrong. If a
seed-to-seed variance number is ever needed, it cannot come from this directory
without first establishing which two of the three actually differ only by seed.

`seed0`'s config is also from an older schema: it is 28,192 bytes against the
others' ~28,410, and it does not record `backbone_target` at all, where the
other two name `TransfuserBackbone` explicitly. Reading a config from
`reference/seed0` and expecting today's keys will not work.

## `checkpoints/transfuser` is the same model again

The README drives its four example commands against
`--checkpoint checkpoints/transfuser`, which is a separate 264 MB directory
outside `reference/`. It is not a separate model. Its config records the same
upstream run as `reference/seed0` --
`001_training/posttrain/260807_133603` -- so the two directories hold one
checkpoint, fetched twice: into `checkpoints/transfuser` on 2026-09-15 and into
`reference/seed0` on 2026-09-25.

That matters on a fresh clone. The README's commands do not work until those
264 MB are back, and nothing in the repository fetches them; whoever restores
this project has to obtain the checkpoint and can then satisfy both paths from
one download.

## What reads them

`ref0` rows appear in four result files:

- `results/reference_closed_loop.csv`
- `results/ref_both_control.csv`
- `results/governor_joint_degradation.csv`
- `results/weather_closed_loop.csv`

Their tracked copies are under `thesis-artifacts/results/`, so the numbers
survive the loss of this directory. The models do not, and neither does the
ability to score the reference on a new route set or a new condition.

## Getting them again

The download source is not recorded anywhere, including here — that is a gap,
not a decision. What is known: they carry the upstream author's paths, the
files are ~276 MB each, and they arrived on 2026-09-25. Anyone rebuilding this
project on a new machine should ask the LEAD authors for the same three
post-train checkpoints rather than guess, and should record the answer in this
file when they get it.
