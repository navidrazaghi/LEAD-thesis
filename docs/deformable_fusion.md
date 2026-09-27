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
this size, and that dense fusion attention is 2.2% of the forward pass anyway —
so there was never much to win. The second half is measured under
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
number which bounds every attempt at making fusion cheaper: optimising a tenth
of the model cannot return more than a tenth.

`scripts/common/forward_profile.py` times each module with CUDA events around
it. Shares of the whole forward pass, batch 8:

| part                | rung0 (dense) | rung4 (deformable) |
| :------------------ | ------------: | -----------------: |
| **all fusion blocks** |     **2.2%** |          **36.6%** |
| all encoder stages  |         15.9% |              10.9% |
| `radar_detector`    |         33.5% |              29.0% |
| `planning_decoder`  |         24.7% |               6.1% |

Three things follow, and the first is the answer to the question this section
opened with.

**Dense fusion attention is 2.2% of the forward pass.** The premise that
attention imposes a significant computational cost does not hold in this
architecture at this token count. Removing it entirely would return two percent.
There is nothing there to reclaim.

**The deformable operator turns that 2.2% into 36.6%** — sixteen times larger,
and the largest single part of the model. That is the whole of the 13% slowdown
in `results/cost.csv`, explained: a negligible part became a third of the work.

**The largest consumer of the baseline is the radar detector, at 33.5%** — a
module no claim in this thesis depends on, evaluated nowhere and part of no rung.
If reducing cost were the goal, that is the obvious target, and it has nothing
to do with attention.

Two caveats. The absolute milliseconds were taken while a training run held the
same GPU, so they are inflated roughly fourfold against the idle-device figure
in `results/cost.csv`; the shares are what the measurement is for. And
`planning_decoder`'s share differs between the two models far more than the
architecture explains, which is likely the same contention and wants a clean
re-run. Raw output is in `results/forward_profile.txt`.

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
was built for.

Reproduce the sweep on the training GPU with

```console
user@host:~/lead$ LEAD_RUNTIME_TYPE_CHECKING=false python scripts/common/bench_fusion_attention.py --device cuda --compile
```

The flag is not optional under `--compile`: beartype and Dynamo cannot run
together, which is why `scripts/common/pretrain.sh` clears it too.

Both `lidar_bev_grid_rows` and `lidar_bev_grid_cols` are `overridable_property`,
so the BEV token grid can be refined from the command line; the image grid is
derived from the camera geometry and needs `avgpool_img` to change with it.
