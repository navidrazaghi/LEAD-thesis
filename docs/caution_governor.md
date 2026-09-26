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

```console
user@host:~/lead$ python scripts/common/run_evaluation.py \
    --models rung4=outputs/rung4_light_auxiliary_post \
    --conditions none:0 both:0.5 both:1.0 \
    evaluation.inference.use_caution_governor=true
```

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

## What has not been measured

No closed-loop run has used the governor. The signals are implemented and
tested, the calibrator is verified on synthetic streams, and the observability
signal's range is measured from recorded runs — but no driving score has been
attributed to any of it. The headline ablation the design is aimed at, the same
signal actuated at attention level against behaviour level, is still ahead.

One practical constraint on getting there: the open-loop pre-screen does not
work on this stack. Ranking checkpoints by waypoint error agrees with the
simulator on 56% of within-condition pairs against 50% for a coin flip, and it
picks the gated rung in every condition while the simulator ranks it middle or
last in every condition. `scripts/common/openloop_prescreen.py validate` reports
this and refuses to endorse its own ordering. Closed-loop budget therefore has
to be spent directly, with fewer conditions rather than fewer routes — the
routes are what give the paired comparison its power.
