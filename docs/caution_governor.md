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
| `observability` | the trained head, best modality per cell        | single-modality failure |
| `cross_modal`   | camera depth against the LiDAR returns          | both modalities wrong the same way |
| `ensemble`      | disagreement between four waypoint readouts     | — |

`observability` costs no training at all. `cross_modal` needs no label either:
two sensors looking at one world have to agree about what is there, and a broken
one stops agreeing. `ensemble` needs a short fine-tune on frozen features and is
the only one that says anything about a scene that is perfectly visible and
simply unlike anything in training.

### The default signal is inert under every condition run so far

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

So the governor on its default rule does nothing under any of the five
conditions this project has run. Its regime is joint degradation — the `both`
condition — where redundancy has nothing left to fall back on. `mean` and
`worst` are configurable so that choice is measured rather than argued.

That regime turned out not to be drivable at all. See
[the result](#the-governor-has-no-leverage-in-this-stack) below before building
on any of this.

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

The chain of reasoning is closed and every link is measured:

- under **single-modality** damage the signal is correctly quiet — one working
  sensor is enough, and `rung2a` scores better under full camera destruction
  than intact;
- under **joint** damage the signal is loud, caution 0.99;
- but no checkpoint drives there, ours at 28% completion or the reference at 3%;
- and slowing a vehicle that is already stuck only lengthens how long it stays
  stuck — 1093 s to a wedge became 2700 s to a timeout.

**The only regime where this signal is informative is the regime where driving
itself collapses.** That is a property of the condition, not of the mechanism:
the governor, its three signals and the calibrator all work as specified and are
tested. They have no lever here.

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

The governor has run closed-loop, once, and the result is the section above: in
the only condition where its signal says anything, no checkpoint drives. The
headline ablation the design was aimed at — the same signal actuated at
attention level against behaviour level, all else equal — cannot be run, because
the behaviour-level arm has no regime to act in.

What that leaves standing, and what it does not:

- The three signals, the actuator and the calibrator are implemented and tested,
  and the ensemble's spread is measured across seven conditions. None of it has
  a driving score attached, and on the evidence here none will without a
  checkpoint that drives under joint degradation.
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
