# The finer token grid

The one direction in which the deformable operator was never given a fair test,
and the design of the rungs that would give it one.

## Why this exists

[Deformable fusion attention](deformable_fusion.md) records a negative result
and the two measurements behind it. The operator computes 69x fewer scores than
dense attention and is 13% slower end to end, because at 552 tokens the dense
path is one fused SDPA kernel and the sparse path is a sequence of small
gather-shaped ops. Profiling then showed there was never much at stake either
way: the fusion blocks are a small part of the dense model's forward pass, and a
large part of the sparse one's.

Both facts are about one geometry. The same benchmark says the operator wins by
3.07x at 2208 tokens and 6.29x at 8832. The stride-32 pooling that produces 552
tokens exists *because* dense attention is quadratic, so the operator is being
judged on a grid shaped by the constraint it removes.

One correction since this was written. The 2.2% is withdrawn: repeating the
profile gives 6.9% and 12.9% for the same quantity, and the point estimate is
not something to build on. It does not change the case for this rung — the
argument needs only that fusion is a small part of the dense model and a large
part of the sparse one, which every run agrees on — but nothing here should be
read as resting on a specific percentage.

The question this document plans is therefore not how to make fusion cheaper.
It is:

> Does a finer fusion token grid drive better, and if so, is the sparse operator
> what makes it affordable?

## What the backbone actually does

Three things had to be read out of the code before the rung could be designed,
and each changed its shape.

**Fusion happens four times, not once.** `fuse_features` is called at the end of
every encoder stage and the backbone builds four separate `GPT` blocks. Whatever
share is attributed to fusion is the sum of all four.

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

These resolutions were derived from ResNet34's geometry when this was written
and have since been measured: the pre-flight reports image maps of 96x288,
48x144, 24x72 and 12x36 and BEV maps of 80x96, 40x48, 20x24 and 10x12, which is
exactly the table above. The cap bites at stage 3 and nowhere else, as designed.

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

### What it found, which was not a verdict

The geometry it confirmed. The timings it refused to conclude from, and the
refusal is the useful part.

Its first run reported a stage with 64 channels as twenty-five times more
expensive than the next stage with 128, which no attention operator does. That
was the first module timed paying for compilation and autotuning, so each
configuration is now timed twice with the first run discarded. The discarded run
sat up to **10.3x** from the kept one, so the discipline was load-bearing.

It was also not sufficient. With it in place, the deformable operator at 552
tokens measured 0.35, 0.58, **8.30** and 1.38 ms at 64, 128, 256 and 512
channels, and one configuration timed twice within a single process came out at
1.13 ms and 9.44 ms. A sanity check now refuses to publish a verdict computed
from timings that invert against channel width, because a script that concludes
confidently from noise is worse than one that stops.

The cause is not our own contention. `nvidia-smi` shows another user's process
resident on the card throughout, and utilisation read 34% at the start of a
window taken immediately after stopping our own training. There is no idle A100
to wait for on this machine.

### The timing method that came out of it

`scripts/common/interleaved_timing.py` is that method, and it is a module rather
than a fix inside the pre-flight because the same problem will come back the
next time anything on this machine is timed.

Two changes. Configurations are **interleaved** in rotating rounds instead of
each being measured to completion, so a burst lands across whatever is running
rather than on whichever one it was scheduled against. And the reported cost is
the **cheapest round**, not the mean or the median: contention only ever adds
time, so of fifteen rounds the cheapest is the one that came nearest to having
the card alone.

The minimum was not the first choice. The module used the median until its own
validation run argued it down -- under live training the medians sat up to 1.6x
above the cheapest round while the minima were clean and monotonic for both
operators.

It checks itself before reporting anything. One configuration is built twice, as
two separate modules of identical shape, and the two costs are compared; they
are the same computation, so a disagreement is the method failing rather than a
result. A second gate refuses if the median round sits more than 4x above the
cheapest, which would mean almost no round ran quiet.

**Validated under the worst conditions available**, with a training run on the
same card (`results/interleaved_timing_under_load.txt`):

| | |
| :--- | ---: |
| two independent copies of one configuration | agree to **8.9%** |
| worst contention seen | median 1.9x the cheapest |
| individual rounds thrown out by taking the minimum | up to **57 ms** against a 0.17 ms cost |

Both operators come out monotonic in channel width, and dense beats deformable
at 552 tokens at every width above 64 -- which is the result already on record,
arrived at independently. The agreement is 8.9% against a 10% tolerance, so it
passed without much room; that is worth knowing rather than rounding away.

### What is still not measurable

The operator half of the pre-flight is fixed. The other half is not.

The prediction also needs the fusion blocks' share of the whole forward pass,
and that comes from `forward_profile.py`, which is the instrument that gave
2.2%, 12.9% and 6.9% for the same quantity.

The cause is not what this document first guessed. Compilation was the suspect,
but `PolicyRunner` loads the policy eager, so the module boundaries the hooks
attach to are real ones. What actually happens is that the driver time-slices
our context away in favour of the other user's, our stream stalls, and the stall
lands inside whichever module's event window was open. The whole forward pass
absorbs every stall wherever it falls, which is why the total held steady while
the parts moved.

Interleaving does not fix that, because the problem is where the time is charged
rather than how much of it there is.

`scripts/common/fusion_cost_by_difference.py` measures it the way that
diagnosis implies: run the whole model, run it again with `fuse_features`
replaced by a pass-through, and subtract. Both measurements are of the entire
forward pass, so nothing has to be attributed to a module at all -- the question
being asked twice is the one that was stable in every earlier run.

It refuses when it cannot resolve. The unmodified model is timed twice, and the
gap between those two is the machine's noise floor; a difference that does not
clear it by a factor of two is reported as unresolved rather than as a small
number. That distinction matters here: "fusion costs less than this machine can
measure" is not the same claim as "fusion costs almost nothing", and the earlier
2.2% was the second dressed as the first.

#### What four runs of it showed, and why none is published

All four ran with our own training on the same card, which turned out to be the
thing that decides the answer.

Interleaving the ablated model against the intact one, round by round, took
rung0's noise floor from 84 ms to 16 ms -- the first version held the
pass-through for a whole run, which made the two measurements separate
contiguous windows and reproduced the original problem one level up. Shortening
the rounds helped far more: at three forwards per round a continuously busy card
contends with nearly every one, and at one forward per round across sixty rounds
the minimum has sixty chances at a quiet slot. That took the floor to 0.37 ms,
and both models resolved -- fusion at 20.5% of rung0's forward pass and 25.7% of
rung4's, at twenty-nine and twenty times the floor.

The next run, identical settings, minutes later, gave 14.2% and 15.5% and
resolved neither. The differences themselves disagree by 49% and 82% between the
two runs.

So the resolved run was luck, and its internal check could not see that: the
intact model and its replica had both caught quiet slots, agreed with each
other, and certified a session in which the ablated model had not. **Two
configurations agreeing does not vouch for a third.** The script now repeats the
whole measurement and requires the differences from the repeats to agree, which
is a check chance cannot pass twice.

`scripts/common/run_idle_measurements.sh` runs it in the handover gap between
training jobs, which is the one condition none of these four runs had.

**No hours are booked for F1 or F2 until it reads ACCEPTED there.** What can be
said today is only that fusion is somewhere in the low tens of percent for both
operators rather than the 2.2% once recorded -- which is enough to retire the
claim that there was nothing to reclaim, and not enough to plan against.

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
