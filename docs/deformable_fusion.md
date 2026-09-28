# Deformable fusion attention

An alternative fusion operator for the TransFuser backbone: each token samples a
fixed number of learned points on each modality's grid instead of scoring
against every token.

## What it replaces

`TransfuserBackbone.fuse_features` pools both branches onto fixed anchor grids
and hands the concatenation to a `GPT` stack, whose `SelfAttention` is dense
over the combined sequence. With the shipped config the grids are

| grid  | source                              | size    | tokens |
| :---- | :---------------------------------- | :------ | -----: |
| image | `384 x 1152` at stride 32           | `12x36` |    432 |
| BEV   | `320 x 384` raster at stride 32     | `10x12` |    120 |
|       |                                     |         |  **552** |

so every fusion block computes `552 x 552` scores per head. The deformable
operator computes `552 x L x K` reads instead — 69x fewer at `L=2, K=4`.

## Selecting it

```console
user@host:~/lead$ python -m lead.training.train \
      policy.transfuser.backbone_target=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone
```

| key                                                    | default | what it does                                     |
| :----------------------------------------------------- | ------: | :----------------------------------------------- |
| `policy.transfuser.deformable_num_points`              |     `4` | points sampled per query, per head, per modality  |
| `policy.transfuser.deformable_learn_cross_reference`   |  `true` | queries refine their own reference points         |
| `policy.transfuser.deformable_calibrated_reference`    | `false` | seed the cross-modal references from the rig      |
| `policy.transfuser.deformable_reference_height_meter`  |   `0.8` | height at which the two grids are corresponded    |

This is a different operator, not a repacking of the same arithmetic, so it
changes what the model computes. Checkpoints do not transfer: the fusion
transformers hold different parameters.

## Reference-point geometry

Deformable DETR's levels are one scene at several scales, so a query's
normalized position transfers across levels unchanged. Here the levels are two
modalities with unrelated geometry — a perspective image grid and a top-down
BEV grid — and the same normalized coordinate denotes different places in each.

A query therefore anchors at its own cell centre on its own modality. On the
other modality it starts from one of two places, and which one is the point of
`deformable_calibrated_reference`.

Off, it starts at that grid's centre and has to learn the correspondence from
scratch. On, `fusion_geometry` supplies the answer the rig already knows: every
BEV cell centre is projected into the stitched image through the cameras'
mounting poses and fields of view, and every image token's ray is cast onto a
horizontal plane `deformable_reference_height_meter` above the ground. With the
shipped rig that covers 296 of the 552 tokens — the rest are BEV cells outside
the three forward cameras, and the top half of the image, which looks at or
above the horizon. Those keep the centre fallback.

Either way the refinement projection is zero-initialized, so training starts
exactly at whichever seed is in use and moves from there. Setting
`deformable_learn_cross_reference=false` freezes the reference points at the
seed instead.

One table serves every sample and every layer. The token grids are fixed by the
backbone's pooling, and the rig is fixed from the model's side: when data
collection perturbates the cameras, the loader re-expresses labels and LiDAR in
the perturbated rig's own frame, so the network always observes the nominal
calibration.

> The projection here does not reuse `lead.common.sensors.camera`. That helper
> applies the mounting rotation without transposing it, which is the identity
> only for the forward camera; on a yawed camera it puts a point on the optical
> axis behind the lens. It feeds visualization overlays only.

## Where the cost actually falls

Two separate questions live here and they have different answers: which operator
is faster, and whether either matters. The short version is that dense wins at
this size, and that how much fusion is worth against the rest of the model
turned out not to be reliably measurable here — the same profile repeated gives
2.2%, 6.9% and 12.9%. What does reproduce is that the sparse operator makes the
fusion blocks a much larger part of the model, and that it costs 12–13% end to
end. Both are under
[How much of the model the operator even is](#how-much-of-the-model-the-operator-even-is).

The score-count ratio is not the wall-clock ratio. At `T=552` the dense path
runs as a single fused SDPA kernel while the sparse path is a sequence of small
gather-shaped ops, so the dense operator is the faster of the two despite doing
far more arithmetic. Measured on one A100-SXM4-40GB, batch 16, `C=256`, under
`torch.compile` — the way training runs:

| stride | tokens | dense | deformable | speedup |
| -----: | -----: | -------: | ---------: | ------: |
| **32** | **552** | **0.77 ms** | **1.39 ms** | **0.55x** |
|     16 |   2208 |  10.77 ms |    3.51 ms | 3.07x |
|      8 |   8832 | 103.59 ms |   16.47 ms | 6.29x |

Stride 32 is the geometry the shipped config produces, and there the sparse
operator *loses*: compilation is worth more to the dense path at that size than
sparsity is. The crossover sits between stride 32 and 16, and past it the gap
widens fast.

So the operator does not pay for itself by being dropped into the grid that is
already there. It pays by letting the grid get finer — the stride-32 pooling
exists because dense attention is quadratic, and relaxing it is what sparsity
buys. Eager numbers differ (deformable is mildly ahead at stride 32, behind at
24), so quote compiled numbers or say which you mean.

### How much of the model the operator even is

The table above compares the two operators against each other. It does not say
what either is worth against the rest of the forward pass, and that is the
number which would bound every attempt at making fusion cheaper: optimising a
tenth of the model cannot return more than a tenth.

That is the question. The answer this section used to give was a set of point
estimates, and they did not survive being measured again.

`scripts/common/forward_profile.py` times each module with CUDA events around
it. Run three times on the same two checkpoints, the share of the forward pass
attributed to the fusion blocks came out as:

| run | conditions | rung0 (dense) | rung4 (deformable) |
| :-- | :--------- | ------------: | -----------------: |
| 1 | a training run held the same GPU | 2.2% | 36.6% |
| 2 | after stopping our training | 12.9% | not taken |
| 3 | after stopping our training | 6.9% | 29.3% |

The whole forward pass is stable across those runs -- 123.54 ms and 122.57 ms
for rung0 in runs 2 and 3 -- while the parts move by a factor of two. The
denominator holds and the numerators do not, which points at the instrument
rather than at the machine's load. `radar_detector` moved from 33.5% to 44.0%
between runs and `planning_decoder` from 24.7% to 13.5%. The likely cause is
that these models run compiled: with `torch.compile` the module boundaries the
hooks are attached to are not necessarily where the work happens, so events can
bracket regions that do not correspond to the module they are named after.

**The point estimates this section used to publish are withdrawn.** "Dense
fusion attention is 2.2% of the forward pass" was quoted in three places in this
repository, and the same measurement repeated gives 6.9% and 12.9%. None of the
three is trustworthy enough to build on, and the claim that followed from it --
that there is nothing to reclaim there -- is not supported by an instrument that
disagrees with itself twofold.

What does reproduce is worth separating out and keeping.

**The direction is stable.** The fusion blocks take a far larger share under the
deformable operator than under the dense one, in every run: 2.2% against 36.6%,
and 6.9% against 29.3%. The operator moves fusion from a small part of the model
to something between a quarter and a third of it.

**The end-to-end cost reproduces independently.** rung4's whole forward pass is
137.19 ms against rung0's 122.57 ms, a 12% slowdown, which agrees with the 13%
recorded in `results/cost.csv` from an entirely separate measurement. That is
the number to quote, because two instruments agree on it.

**The radar detector is large in every run.** It is the biggest single consumer
of the dense baseline at 33.5% and again at 44.0%. The share is not reliable but
the ordering is, and the observation that follows does not need a precise
number: no claim in this thesis depends on that module, it is evaluated nowhere,
and it is part of no rung. If reducing cost were the goal, it is a better target
than attention.

**The card is shared.** `nvidia-smi` shows another user's process resident
throughout, and utilisation was 34% at the start of a window taken immediately
after stopping our own training. There is no idle A100 here to retreat to, which
is why the fix for this measurement is a method robust to contention rather than
better scheduling.

Raw output from the most recent run is in `results/forward_profile.txt`.

### What follows from this

The operator is not used by work after the thesis. `scripts/common/run_dense_line.sh`
trains on the dense path, and the reasoning is the two measurements above: there
was nothing to reclaim at 552 tokens, and the sparse operator costs 13% to find
that out.

That switch is not free, and the price is a control. Every rung of the thesis
ladder except rung0 is deformable, and rung0 has no degradation curriculum, so a
dense rung has nothing to be compared against on equal terms. The dense line
therefore opens with `rung2a_dense_curriculum` — dense plus the curriculum,
nothing else — before any rung that asks a new question.

It is worth more than its overhead. Against rung2a it isolates the operator
under the curriculum, and against rung0 it re-measures the curriculum on the
dense path, which says whether the thesis's one solid positive result ever
depended on the operator at all.

Two things do not carry across. The observability head does: it hangs off the
top-down pyramid rather than the fusion operator, and the dense backbone builds
that pyramid whenever `use_observability` is set. The gate does not, and says so
rather than failing quietly — `Transfuser.__init__` raises, because the gate
biases a softmax over a modality axis that only the deformable operator has.
Since the gate is the thesis's negative result, that closes nothing that was
open.

Nothing here retracts the operator itself. The stride-16 and stride-8 rows above
stand, and they are what would make it worth returning to: a finer token grid is
the condition under which sparsity pays, and it is the direction the operator
was built for. That direction is planned in [The finer token grid](finer_grid.md),
with a pre-flight that can refuse it before a GPU is booked.

Reproduce the sweep on the training GPU with

```console
user@host:~/lead$ LEAD_RUNTIME_TYPE_CHECKING=false python scripts/common/bench_fusion_attention.py --device cuda --compile
```

The flag is not optional under `--compile`: beartype and Dynamo cannot run
together, which is why `scripts/common/pretrain.sh` clears it too.

Both `lidar_bev_grid_rows` and `lidar_bev_grid_cols` are `overridable_property`,
so the BEV token grid can be refined from the command line; the image grid is
derived from the camera geometry and needs `avgpool_img` to change with it.
