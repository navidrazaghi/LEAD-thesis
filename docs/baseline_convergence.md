# The baseline that never converged, and what fixing it revealed

Two findings, and the second was not the one being looked for.

## The baseline was a diverged run

Every comparison in the thesis is measured against `rung0_baseline`. Its
training loss reaches a floor at epoch two and rises from there, ending 2.5x
above its best in every tracked head:

```
semantic  0.103  0.067  0.051  0.071  0.150  0.117  0.110  0.114  0.184  0.151
                        ^ floor  ^ leaves it and does not come back
```

Seven of the nine trained rungs converge on the same data and the same epoch
budget. Two do not: this one and `rung2b_observability_ungated`. Measured from
the offline W&B logs, final over best is 3.36x and 2.14x on the training
objective, and 2.95x and 2.83x on the unscaled semantic head, against
1.00-1.04x for every other rung — so it is not one unlucky seed.

An earlier version of this note named `rung2a_dense_curriculum` as the second,
reasoning that it is the only other dense run. That reasoning explained a
pairing instead of measuring it, and the measurement does not support it: the
dense curriculum run ends at 1.20x, elevated but far from the two above.
`scripts/common/training_curves.py` extracts the series, and the thesis plots
them.

### The cause was the recipe, not the architecture

The architecture is LEAD's and LEAD converges with it. Four things were changed
from what LEAD ships and the learning rate was not one of them:

| | LEAD | this project |
| :--- | ---: | ---: |
| batch size | 64 | 8 |
| epochs | 31 | 10 |
| data | full | 450 logs |
| learning rate | 3e-4 | 3e-4 |

An eighth of the samples per step is about 2.8x the gradient noise, by
`SE = sigma/sqrt(B)`. A step size tuned for the quieter gradient, applied to the
noisier one, is the ordinary way to get a loss that finds a floor and then leaves
it.

### Restoring the recipe fixed it

`scripts/common/run_baseline_lead_recipe.sh` trains `rung0_lead_recipe` with the
effective batch restored to 64 through gradient accumulation -- 32 real samples,
accumulated twice -- so the shipped learning rate needs no rescaling and no
rescaling has to be defended. Measured, that peaks at 23.5 GB against 37.3 GB for
a true batch of 64, which matters on a card shared with another user.

The data was deliberately left alone. Enlarging the subset in the same run would
have left nothing able to say which change did the work.

The curve descends monotonically for all 31 epochs and reaches its minimum at
the last one:

```
semantic  0.136  0.063  0.045  0.039  0.035  0.031  0.028
final/best = 1.00x     (the old run: 2.95x)
```

Both ratios here are on the semantic head, matching the top of this note.
The figure 2.52x that stood here until 2026-10-04 predates
`training_curves.py`: with the curves extracted, no combination of the
logged heads reproduces it, and the measured value is 2.95x.

The final semantic loss is **5.4x lower** than the run it replaces, and lower
than `rung2a`'s 0.043 -- the best sparse rung. No extra data was needed; the
disk freed earlier in the week went unused.

## And then it drove worse -- withdrawn, the two sides were not scored alike

This was the finding that was not being looked for, and it does not survive
checking how the second half of it was measured.

> **Correction.** The claim below rests on the two runs having been "scored on
> the same protocol". They were not. The protocol caps each route at
> `_ROUTE_TIMEOUT_S` of *wall clock*, and a route killed at the cap returns no
> score at all. The original baseline was scored on an idle machine and put one
> route at the cap. The retrained baseline is being scored while another user
> holds seventeen of the thirty-two cores: measured directly from
> `/proc/<pid>/task/*/schedstat`, CARLA spends **61%** of its ready time waiting
> for a CPU and the agent **54%**, and the median wall time per route has gone
> from **475 s to 1314 s**. Ten of its first twenty-nine routes are at the cap.
>
> So the timeout rates below are not comparable: one side is partly a
> measurement of who else was on the machine. `scripts/common/censored_analysis.py`
> now bounds the comparison instead of asserting it, and on the intact condition
> the direction of the difference **flips between the bounds** -- this data
> cannot settle whether the retrained model drives better or worse.
>
> The training-curve half of this document is unaffected. Loss values do not
> depend on wall clock.

The uncorrected observation, kept for the record:

Scored on what was believed to be the same protocol, the retrained baseline
times out on **36%** of routes against **3%** for the diverged run it replaces
and **0%** for `rung2a`.

Every open-loop measure says it is the better model:

| | old baseline | **retrained** | rung2a |
| :--- | ---: | ---: | ---: |
| semantic loss | 0.151 | **0.028** | 0.043 |
| stop rate vs expert | 2.51x | **1.86x** | 1.80x |
| route ADE (m) | 0.701 | **0.127** | 0.324 |
| waypoint ADE (m) | 0.691 | **0.458** | 0.442 |
| **route timeouts** | 3% | **36%** | 0% |

Four open-loop measures, all in one direction. Closed-loop behaviour in the
other.

### Three explanations were measured and refused

Each was plausible and each is now ruled out, which is worth recording so they
are not proposed again.

**It plans to drive more slowly.** Refused. `plan_under_damage.py` measures the
displacement between consecutive predicted waypoints: 1.899 m intact against
1.893 m with the lidar destroyed, a 0.3% difference where the score gap is 21%.

**It is the inertia problem** -- the spurious correlation between being stopped
and staying stopped that Codevilla documents. Refused, and in the opposite
direction: `inertia_probe.py` finds the retrained model picks the stop class
26.7% of the time against the old model's 36.0%, on identical frames, where the
expert picks it 14.4%. Longer training reduced stopping. CILRS's remedy --
predicting target speed as an auxiliary task -- is already in this stack and is
working.

**It has learned a speed ceiling.** Refused: the ceiling is the data's. The
expert's own labels put 79% of frames in the 8 m/s class and never use the five
classes above it. The model reproduces the label distribution it was shown.

### What it leaves

No open-loop measure explains the closed-loop behaviour, and that is the result
rather than a gap in it. It is the open-loop/closed-loop divergence the thesis
argues for on theoretical grounds -- the state distribution depends on the policy
itself, so imitating the expert's distribution more exactly guarantees nothing
about the distribution the policy visits -- observed directly and quantitatively,
in a model that is better by every imitation metric available.

One hypothesis remains untested and is stated as a hypothesis: the retrained
model follows the reference route far more tightly (route ADE 0.127 against
rung2a's 0.324), and tight following may leave it less able to recover once it is
somewhere the expert never was. Nothing here measures that yet.

## Status and caveats

The closed-loop evaluation is running and was at 22 of 90 routes when these
numbers were taken, all in the intact condition. The timeout rate may move as the
degraded conditions land. Nothing in the second half of this document should be
treated as settled until it finishes.

The first half -- the divergence, its cause, and that restoring the recipe fixes
it -- rests on training curves rather than on driving scores and does not depend
on the evaluation completing.

## What this obliges the thesis to do

The baseline every comparison is measured against never converged, so the
comparisons inherit that. The correction is not necessarily favourable: a
baseline that trains properly scores higher, which narrows every margin reported
over it.

`docs/thesis_revisions.md` lists what needs rereading once the evaluation
finishes. The finding itself -- that the baseline diverged, how it was
diagnosed, and what fixed it -- belongs in the thesis rather than only here. An
examiner who finds it first is a worse outcome than one who reads it in the
methodology.
