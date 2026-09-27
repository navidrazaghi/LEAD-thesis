# The caution governor

Actuating an uncertainty signal in the controller instead of the attention
logits, with the conservativeness calibrated online rather than tuned.

## Why

The observability gate is informative and does not drive better. The ladder
measured both halves of that: the gate moves attention mass by a factor of
eleven to thirty-eight in the right direction, and it does not improve the
closed-loop driving score — under full camera destruction it costs 15.1 points
with an interval clear of zero.

The measured reason is where the signal was wired. The gate acts on attention
logits, and attention's authority over the token leaving a fusion block is about
four tenths, so a large shift in what the model reads became a small shift in
what it does. The decomposition holds for both gated rungs: attention mass at
BEV queries × the attention's residual authority reproduces the causal reliance.

The governor takes the same signal to a place where the authority is not
fractional. If the model cannot resolve the road ahead, the car slows down.
Nothing about the network changes — the checkpoint is frozen, the forward pass
untouched, and the only thing read is a head the model already computes.

## Three signals, and what each is blind to

| signal          | reads                                          | blind to |
| :-------------- | :--------------------------------------------- | :------- |
| `observability` | the trained head, best modality per cell        | LiDAR failure (see the sweep) |
| `cross_modal`   | camera depth against the LiDAR returns          | both modalities wrong the same way |
| `ensemble`      | disagreement between four waypoint readouts     | single-modality failure |

`observability` costs no training at all. `cross_modal` needs no label either:
two sensors looking at one world have to agree about what is there, and a broken
one stops agreeing. `ensemble` needs a short fine-tune on frozen features and is
the only one that says anything about a scene that is perfectly visible and
simply unlike anything in training.

Two of those blind spots were claimed from the design and then measured. One
claim held and one did not.

### What each signal actually does, measured

Swept over cached frames, identical damage and identical frames for each, by
`caution_forward_sweep.py` and `ensemble_spread_range.py`. Swing is against the
same signal's own intact reading, because the three do not share a scale.

| condition       | `observability` | `ensemble` | `cross_modal` |
| :-------------- | --------------: | ---------: | ------------: |
| camera 0.5      |          +0.155 |     +0.005 |        +0.050 |
| **camera 1.0**  |      **+0.175** |     +0.048 |    **+0.225** |
| lidar 0.5       |          -0.074 |     -0.017 |        -0.008 |
| lidar 1.0       |          -0.186 |     -0.055 |        -0.042 |
| both 0.5        |          +0.206 |     -0.039 |        +0.058 |
| **both 1.0**    |          +0.431 |     +0.323 |    **+0.267** |

Intact baselines, which the swings sit on top of and which are not evidence of
anything: 0.390, 0.267 and 0.608.

**This corrects an earlier claim in this document.** The section below reported
`observability` as inert under every single-modality condition, swinging 0.017.
That number came from `caution_signal_range.py`, which reads the recorded
per-modality means over the whole BEV grid — not the quantity the governor
computes, which takes the better modality *per cell* and then averages over the
driving corridor alone. Measured properly, on the same frames as the others,
the swing under full camera destruction is 0.175. The proxy was off by an order
of magnitude and the conclusion drawn from it was wrong.

What survives that correction, and what does not:

- `cross_modal` does react to a single failed sensor, and most strongly of the
  three. The claim behind it holds — a signal that *compares* two sensors sees
  one of them break.
- `observability` reacts too, which the design said it would not. Taking the
  better modality per cell does not hide a failed camera as completely as the
  grid-wide proxy suggested.
- The `ensemble` claim did not hold. It was built to cover single-modality
  failure and is nearly silent there, waking only under joint degradation. That
  makes it an out-of-distribution detector whose notion of "out" is set by the
  training curriculum: since the curriculum only ever damaged one modality,
  single-modality damage is *inside* the distribution and the members agree.

One pattern is shared by all three and none of them was designed for it: every
signal reports LiDAR degradation as a *negative* swing. Thinning the sweep
removes evidence rather than adding contradiction, so each of these gets
quieter as the LiDAR gets worse. None of them is a LiDAR-fault detector.

Three more things about `cross_modal` have to be said next to the good number.

**It is asymmetric, and one direction is backwards.** It sees the camera fail
(+0.225) and reports LiDAR failure as a small *improvement* (-0.042). The
definition explains it: with the sweep thinned, occupied cells disappear, so the
contradiction it counts most readily — LiDAR says occupied, camera sees through
— fires less often. It is a camera-fault detector more than a cross-modal one.

**Its intact baseline is 0.61.** On a clean scene it calls three fifths of the
corridor contradictory, which means most of what it counts is modelling error
rather than sensor fault: the 2.5 m tolerance, the single reference height every
cell is projected at, and the depth head's own error. The swing rides on top of
a large offset, and the offset is not evidence of anything.

**Its noise is still comparable to its signal.** Per-batch standard deviation is
about 0.11 against a swing of 0.225 — better than the ensemble managed, and
still not enough for a per-tick decision. It needs the same smoothing.

The rows are in `results/cross_modal_sweep.csv` and
`results/ensemble_spread.csv`.

### Why this does not simply rescue the governor

The result below is that joint degradation is undrivable, and `cross_modal` is
the one signal that speaks outside it. That is worth stating, and so is the
tension it runs into: the condition it detects best, full camera destruction, is
one the curriculum rung already drives *better* in than intact — 46.0 against
41.8. A governor that slows down there spends route completion to buy nothing.

So the open question this signal inherits is not whether it can see a fault. It
can. It is whether any condition exists where the model both still drives and is
genuinely hurt. The deployment perturbation families are the place to look,
which is why the next rung trains on them rather than on more governor.

### A whole-grid proxy, and why it misled

This is the finding that decides how the governor can be evaluated, so it
belongs here rather than in a commit message.

`observability` combines the two modalities by taking the better of them per
cell. That is what the evidence supports — the trained rungs score no worse
under full camera destruction than intact, so one working sensor is enough and
slowing down for a single failed one would spend route completion to buy
nothing. The consequence is that with one modality destroyed the other still
resolves the scene and the signal correctly reports that nothing is wrong.

Measured on the recorded per-modality means, `scripts/common/caution_signal_range.py`:

| rule    | intact | camera destroyed | swing  |
| :------ | -----: | ---------------: | -----: |
| `best`  |  0.035 |            0.052 | +0.017 |
| `mean`  |  0.060 |            0.474 | +0.413 |
| `worst` |  0.086 |            0.895 | +0.810 |

Read on a whole-grid mean this looks like a governor that does nothing on its
default rule. **That reading is wrong**, and the table in the section above
supersedes it: measured as the governor actually computes it — better modality
per cell, averaged over the driving corridor rather than the grid — the swing
under full camera destruction is 0.175, not 0.017. The grid-wide mean is
dominated by cells the corridor never touches.

The table is kept because the *comparison between rules* it makes is still
informative, and because the mistake is worth leaving visible: a proxy that is
cheap and nearly right in form can be an order of magnitude wrong in size, and
nothing about it announces that.

The conclusion this section originally drew — that the governor's only regime is
joint degradation — did not survive. What did survive is that joint degradation
is undrivable; see
[the result](#the-governor-has-no-leverage-in-this-stack) below.

### The governor has no leverage in this stack

Three single-route closed-loop runs, on route 18305 under `both:1.0`:

| checkpoint            | governor | driving score | route completion | outcome                |
| :-------------------- | :------- | ------------: | ---------------: | :--------------------- |
| `rung4` + ensemble    | on       |             — |                — | timed out at 2700 s    |
| `rung4` + ensemble    | off      |          5.91 |            27.9% | blocked at 1093 s      |
| published reference   | off      |          2.01 |             3.1% | blocked at 1448 s      |

The rows are in `results/governor_joint_degradation.csv`.

The reference scores 90.7 intact and completes three percent of this route with
both sensors destroyed. Nothing drives in this condition, so there is nothing
for a speed governor to act on.

One route is not a measurement of anything, and no claim here rests on the
scores themselves — the design's minimum detectable difference is 17 to 23
points. What the three runs establish is categorical rather than quantitative:
every checkpoint available, with and without the governor, fails to complete the
route.

The chain of reasoning, with the correction above folded in:

- under **joint** damage every signal is loud;
- but no checkpoint drives there, ours at 28% completion or the reference at 3%;
- and slowing a vehicle that is already stuck only lengthens how long it stays
  stuck — 1093 s to a wedge became 2700 s to a timeout.

Under **single-modality** damage two of the three signals do speak, which is not
what this document said before the sweep. That does not rescue the governor, for
a different reason and one the ablation already supplies: `rung2a` scores 46.0
under full camera destruction against 41.8 intact. The model is not hurt there,
so slowing it spends route completion to buy nothing. A signal that correctly
detects a fault the policy is already handling is not a reason to act.

**So there is no condition in this project's matrix where the governor both has
a signal and has something to gain from it.** Where the signal is loud and the
fault is real, nothing drives; where the fault is detectable and driving
continues, the policy is already coping. That is a property of the conditions,
not of the mechanism: the governor, its three signals and the calibrator all
work as specified and are tested.

This is consistent with what the ablation already reported from the other side.
Robustness did not generalise outside the training corruption family, and the
curriculum only ever damaged one modality at a time — by design, so the gate it
was built for had somewhere to shift to. Joint degradation is outside that
family for every model in this repository.

One incidental observation, from a single route and therefore no basis for a
claim: the curriculum-trained rung completed 27.9% of the route against the
reference's 3.1%, despite the reference being three times better intact.

Where this would become worth revisiting: a checkpoint that still drives under
joint degradation, or a condition that degrades both modalities gently enough to
leave the policy driving while still moving the signal. Measured on cached
frames, `both:0.5` does not — its caution sits at zero.

## How much slowing: calibrated, not tuned

A hand-set threshold would be one more knob tuned on the routes it is then
scored on. Instead one scalar rises when a surrogate risk shows up and falls
when it does not:

```
lambda <- clip(lambda + step * (risk - target), 0, ceiling)
```

Its long-run realised risk rate converges to the target from any start, and it
needs no model of how risk depends on caution — only the sign of the error — so
it cannot be wrong about a relationship nobody has measured. What it does not
give is a per-frame guarantee: the bound is on the average over a run. That is
why the mapping keeps a floor under the speed rather than letting the calibrator
stop the car.

The surrogate risk is carrying speed into a stretch the model cannot resolve.
Infractions are the thing worth avoiding, but they are far too rare and far too
late to calibrate on — a run produces a handful, each arriving after the
decisions that caused it.

`tests/unittests/evaluation/test_conformal.py` drives the rule with synthetic
risk streams and checks the realised rate lands on the target from either side.

## Turning it on

Config for an evaluation run travels behind `--config`, not as bare arguments:
`python -m lead` is argparse-only and rejects `key=value`, so the harness
forwards these into the child's config dotlist.

```console
user@host:~/lead$ python scripts/common/run_evaluation.py \
    --models rung4=outputs/rung4_light_auxiliary_post \
    --conditions none:0 both:0.5 both:1.0 \
    --out results/governor.csv \
    --config evaluation.inference.use_caution_governor=true
```

Pass only real knobs there. `evaluation.save_path` looks like one and is a
derived property fed by an environment variable; overriding it raises while the
child builds its config, before the driving agent exists, which the leaderboard
reports as `Agent couldn't be set up` with no traceback in the sweep's log. A
twenty-route run failed that way twenty times before the cause was found.
`tests/unittests/config/test_overridable_knobs.py` now checks the keys every run
script passes, so the same mistake fails in a second instead of an hour.

| key                                                | default          | what it does                                  |
| :------------------------------------------------- | :--------------- | :-------------------------------------------- |
| `evaluation.inference.use_caution_governor`        | `false`          | scale target speed by the caution             |
| `evaluation.inference.caution_signal`              | `observability`  | which signal drives it                        |
| `evaluation.inference.caution_modality_rule`       | `best`           | how the two modalities combine                |
| `evaluation.inference.caution_speed_floor`         | `0.4`            | slowest the governor may drive                |
| `evaluation.inference.caution_target_risk`         | `0.05`           | long-run surrogate risk rate to converge to   |
| `evaluation.inference.caution_initial_lambda`      | `0.0`            | where the scalar starts each route            |

With the scalar at zero the governor is inert and the policy drives exactly as
the frozen checkpoint does, which keeps the un-governed model inside the family:
the headline ablation then compares a mechanism against its own absence rather
than against a different tuning.

## Calibration

The scalar's starting point and the step size that moves it have to come from
somewhere. If they came from the routes the governor is scored on, the mechanism
would be fitted to its own test. `src/lead/routes/eval_sets/calibration.txt`
holds twenty clear-weather routes the scored sets do not use; the selector
builds it by exclusion and verifies the disjointness rather than trusting it.

One property of the data to know about: that set spans two of the eleven towns
its pool offers. The clear pool is fifty routes of Town12 against two to four of
each small town, so once the scored set has taken the small towns a disjoint set
has nothing left to spread over. The selector says so on every run.

## The ensemble signal

Four readouts of the same frozen planning context, each with its own query,
decoder layer and linear head. Three decisions keep it from reporting a number
that means nothing:

- **Its members get a decoder layer, not a bare linear head.** A linear readout
  over identical inputs under an identical loss has one least-squares answer;
  every member finds it and the spread measures nothing.
- **Each member sees a bootstrap resample of every batch.** Fitted to the same
  data from different starts they converge on each other, and the signal decays
  over training while still reporting.
- **It is read in evaluation mode.** The decoder layers carry dropout, so in
  training mode even members with identical weights return different plans — a
  different mask per forward, not a different opinion. Sampling one network's
  dropout is a real uncertainty method and a different one.

Fit it on a finished rung, with that rung frozen:

```console
user@host:~/lead$ bash scripts/common/run_ensemble_finetune.sh \
    outputs/rung4_light_auxiliary_post
```

That freezes 67.7M parameters and trains 3.2M. Then check the signal moves
before spending closed-loop time on it:

```console
user@host:~/lead$ python scripts/common/ensemble_spread_range.py \
    --models rung4_ens=outputs/rung4_light_auxiliary_post_ensemble
```

Read the swing column rather than the spread column. A spread that is small
everywhere and moves nowhere is a collapsed ensemble reporting confidence it has
not earned, and to a governor that is indistinguishable from a model that is
right.

## Status, and what has not been measured

The governor has run closed-loop, once. The headline ablation the design was
aimed at — the same signal actuated at attention level against behaviour level,
all else equal — has not been run, because no condition in this project's matrix
gives the behaviour-level arm both a signal and something to gain from it.

What that leaves standing, and what it does not:

- All three signals are now swept across seven conditions on cached frames, and
  the sweep corrected a claim this document previously made from a cheaper
  proxy. The actuator and the calibrator are implemented and tested. None of it
  has a driving score attached.
- What is missing is a condition, not a mechanism: one where the policy still
  drives and is genuinely hurt. Full camera destruction is detectable and not
  harmful; joint degradation is harmful and not drivable.
- The **calibration route set** is built and verified disjoint from the scored
  sets. It has never been run, because a calibration pass over a condition
  nothing drives in would record twenty routes of nothing.
- The **conformal loop** was corrected before any of this: with the original
  risk speed, slowing to the floor could not reach the threshold at any realistic
  driving speed, so the scalar pinned at its ceiling and the calibration was a
  slow switch. The loop-closure test now guards that.

One practical constraint on getting anywhere near a driving score:

One practical constraint on getting there: the open-loop pre-screen does not
work on this stack. Ranking checkpoints by waypoint error agrees with the
simulator on 56% of within-condition pairs against 50% for a coin flip, and it
picks the gated rung in every condition while the simulator ranks it middle or
last in every condition. `scripts/common/openloop_prescreen.py validate` reports
this and refuses to endorse its own ordering. Closed-loop budget therefore has
to be spent directly, with fewer conditions rather than fewer routes — the
routes are what give the paired comparison its power.
