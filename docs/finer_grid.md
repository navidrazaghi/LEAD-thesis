# The finer token grid

The one direction in which the deformable operator was never given a fair test,
and the design of the rungs that would give it one.

## Why this exists

[Deformable fusion attention](deformable_fusion.md) records a negative result
and the two measurements behind it. The operator computes 69x fewer scores than
dense attention and is 13% slower end to end, because at 552 tokens the dense
path is one fused SDPA kernel and the sparse path is a sequence of small
gather-shaped ops. Profiling then showed there was never much at stake either
way: dense fusion attention is 2.2% of the forward pass.

Both facts are about one geometry. The same benchmark says the operator wins by
3.07x at 2208 tokens and 6.29x at 8832. The stride-32 pooling that produces 552
tokens exists *because* dense attention is quadratic, so the operator is being
judged on a grid shaped by the constraint it removes.

The question this document plans is therefore not how to make fusion cheaper.
It is:

> Does a finer fusion token grid drive better, and if so, is the sparse operator
> what makes it affordable?

## What the backbone actually does

Three things had to be read out of the code before the rung could be designed,
and each changed its shape.

**Fusion happens four times, not once.** `fuse_features` is called at the end of
every encoder stage and the backbone builds four separate `GPT` blocks. The 2.2%
already measured is the sum of all four.

**The anchor grid is a deliberate discard, not an interpolation.** Each stage
hands `fuse_features` its own feature map, and `avgpool_img` flattens it to the
same 12x36 anchor grid every time. With a 384x1152 input and ResNet34:

| stage | stride | image map | what pooling discards |
| ----: | -----: | :-------- | :-------------------- |
| 0 | 4 | 96x288 | 8x8 blocks |
| 1 | 8 | 48x144 | 4x4 |
| 2 | 16 | 24x72 | 2x2 |
| 3 | 32 | 12x36 | **nothing** |

So a finer grid recovers real detail at the first three stages and *upsamples*
at the fourth: more tokens, more attention, no new information. Each stage's
grid therefore has to be capped at that stage's own resolution. An
implementation that ignored the cap would pay the cost at stage 3 and get none
of the benefit, and a prediction that ignored it would overstate both.

These resolutions follow from ResNet34's geometry rather than from a
measurement. `scripts/common/finer_grid_preflight.py` prints the real ones,
which is the first thing to check.

**The cache is not involved.** `_CACHE_FINGER_PRINT_FIELDS` covers raster
geometry and label construction — pixels per metre, extents, box and semantic
label shapes — and not the token grid. The anchor grid is a pooling decision
applied to features the encoders have already produced, so refining it needs no
cache rebuild. That is the difference between a few days and impossible.

## What it would cost

Doubling both grids gives image 24x72 = 1728 and BEV 24x20 = 480, which is 2208
tokens: exactly the stride-16 row already measured. Per block, on an
A100-SXM4-40GB at batch 16, compiled:

| | today, 552 | finer, 2208 |
| :--- | ---: | ---: |
| dense | 0.77 ms | 10.77 ms |
| deformable | 1.39 ms | 3.51 ms |

Under the cap, three stages go finer and stage 3 stays where it is:

* dense: 3 x 10.77 + 0.77 = **33 ms**, against 3.1 ms today
* deformable: 3 x 3.51 + 1.39 = **12 ms**, against 5.6 ms today

The honest reading is not the one that flatters the operator. **The finer grid
is not free under either operator.** Sparsity does not make it cheap, it makes
it affordable. The claim available here is not that the model gets faster than
it is today — it does not — but that an operator which loses at the shipped
geometry unlocks a geometry the dense operator cannot pay for, *if* that
geometry is worth having.

Whether it is worth having is the part nobody has measured.

## The rungs

Two, and the order is the opposite of what the motivation suggests.

**F1: finer grid, dense.** Trained first, even though dense is the expensive
operator here. The question underneath everything is whether the finer grid
drives better, and dense answers it with no operator confounded into it. If the
answer is no, the line ends and the sparse rung is never trained — forty hours
saved by asking the cheaper question first.

**F2: finer grid, deformable.** Only if F1 says the geometry is worth something.
Then this measures whether sparsity is what makes it payable.

Three of the four cells of the design already exist or are queued:

| | dense | deformable |
| :--- | :--- | :--- |
| **552 tokens** | `rung2a_dense_curriculum` *(queued)* | `rung2a` |
| **2208 tokens** | **F1** | **F2** |

The claim the square supports is an interaction: the operator's value depends on
the geometry, with the crossover measured in wall clock and in driving score
rather than asserted from an asymptotic argument. That is a stronger statement
than "my attention is faster", and it is one this project is unusually well
placed to make, having measured both the loss at 552 and the win at 2208.

## What has to change in the code

**The image grid cannot be overridden.** `img_vert_anchors` and
`img_horz_anchors` are plain properties dividing by a hardcoded 32. Setting them
from the command line does nothing — the same trap that killed a calibration run
when `evaluation.save_path` turned out to be a derived property, and the reason
`tests/unittests/config/test_overridable_knobs.py` exists. They need to become
`overridable_property`, or better, to derive from one `fusion_anchor_stride`
knob. The BEV grid is already overridable.

**Pooling has to become per-stage.** `avgpool_img` and `avgpool_lidar` are single
modules shared by all four calls. They become `ModuleList`s, each with the grid
`min(requested, that stage's own resolution)`.

**The positional embedding is sized by the token count.** `GPT` builds `pos_emb`
as `(1, image_tokens + bev_tokens, n_embd)` from the global config, assuming all
four stages share a grid. With per-stage grids it has to take its stage's token
counts. The change is small; the consequence is not — **the parameter shape
changes, so no checkpoint transfers and both rungs train from scratch.**

**The sparse variant needs per-stage reference tables.** `fusion_geometry` builds
the camera-to-BEV correspondence from the grid, so four grids mean four tables.
Bounded work, but not free.

## Before any of it: the pre-flight

`scripts/common/finer_grid_preflight.py` predicts the cost without the
architecture change and without training. It measures the real per-stage
resolutions and channel widths, times both operators at each stage's own
`(tokens, channels)` — today's grid and the finer one — and turns the result
into a share of the whole forward pass.

Timing today's grid as well as the finer one is not redundant: the difference
between a fusion block's measured time and its attention's measured time is what
the rest of the block costs, which is what lets the prediction account for the
feed-forward growing linearly while attention grows quadratically.

```console
user@host:~/lead$ LEAD_RUNTIME_TYPE_CHECKING=false python scripts/common/finer_grid_preflight.py \
      --model outputs/rung0_baseline_post --anchor-stride 16 --compile
```

Run it on an idle GPU. It prints a verdict against one bar: if dense fusion at
the finer grid is still under 15% of the forward pass, a three-times-cheaper
operator returns almost nothing end to end and F2 is not worth its hours. F1 may
still be, because it answers the geometry question rather than the cost one.

## The risks, stated before the result

**The training set may not support the capacity.** Four times the tokens on the
same 450-log subset could buy overfitting rather than resolution. This changed
recently and in the right direction: deleting an unused 239 GB tree took the
disk from 14 GB free to 252 GB, so a substantially larger subset is affordable
now. If F1 comes out worse, that has to be re-run with more data before
concluding anything about geometry — otherwise the result is about the training
set size and says so in the wrong words.

**The benefit may be at the wrong end.** The finer grid buys nothing at stage 3,
which carries the most semantic weight, and most at stage 0, whose features are
the least abstract. It is possible the recoverable detail is not the kind the
planner can use. That is a worthwhile ablation of its own and is deliberately
kept out of the first rung.

**The prediction assumes everything outside fusion is unchanged.** True for the
encoders, whose work happens before pooling. Not necessarily true for
`torch.compile`, which may fuse differently once the shapes change. The step-time
measurement, not the forward-pass prediction, is what settles the training cost.
