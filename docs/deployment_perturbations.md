# Deployment perturbations

Four failure families a deployed stack meets that the appearance curriculum does
not contain, plus the joint-degradation condition used to evaluate them.

## Why

The degradation curriculum already in the repository damages how a sensor
*renders* the world: dimmer, blurrier, noisier images, thinner LiDAR sweeps. It
is the single largest win the ablation ladder recorded, and it generalises
poorly — every trained rung fell below the untouched baseline under LiDAR
corruptions outside the training family.

A deployed stack also fails in ways that leave every rendering untouched. A lens
is blocked and the rest of the frame is pristine. The localisation drifts and
the images are perfect. Perception and actuation cost time, so the waypoints
applied at tick *t* were computed from the observation at *t−k*. None of that is
in the recorded data and none of it is in the existing curriculum.

## The four families

| family        | what it does                                             | where it lives |
| :------------ | :------------------------------------------------------- | :------------- |
| `occlusion`   | blacks out rectangular image regions                      | collated batch |
| `ego_state`   | jitters the navigation points, under-reports the speed    | collated batch |
| `latency`     | re-anchors the planning label onto a later tick           | dataloader     |
| `frame_freeze`| holds a stale observation for a burst                     | evaluation only |

### Occlusion reports what it removed

Every other family scales the observability targets by `1 − severity`, which the
module documents as a modelling assumption rather than a measurement: the
dataset records visibility under nominal sensors only, so the degraded value
cannot be observed. Occlusion is the exception. The mask says exactly what
fraction of the frame survived, so the camera's targets are scaled by a number
that was measured rather than posited.

### Ego state is deliberately outside the modality split

A drifting GPS fix does not make either sensor resolve the scene any less well,
so nothing in this family touches the observability targets. Scaling them would
teach the gate to shift modality on a fault that has no modality.

### Latency is a label transform, not an observation delay

The obvious way to inject latency is to feed the model an older observation.
That is the wrong way *here*, for a reason specific to this repository: the
LiDAR raster is cached and the cache is fingerprinted by `past_lidar_tick_ages`,
so a delayed sweep forces a full re-cache of the training set.

The transform does the same job from the other side. Keep the observation at the
anchor and move the *label* to where the plan will be executed: re-anchor the
future poses onto the pose *k* ticks ahead and take the planning horizon from
there. Only the planning targets are touched, and those are never cached.

Re-anchoring rather than slicing is what makes it exact. A slice would hand the
model a plan expressed around a pose it has already left — a constant offset it
would learn to add back, which is a systematic lag, which is the artefact the
transform exists to remove. `tests/unittests/policy/transfuser/test_latency_curriculum.py`
pins that down: on a straight run at constant speed the shifted label must equal
the unshifted one, and a plain slice misses by more than a metre.

### Frame freeze has no separate training form

A frozen sensor at age *a* is an observation delayed by *a*. Per sample the two
are one transform, so a policy trained across the latency ages already covers
it. What differs is correlation across consecutive ticks, which a single-frame
dataset cannot express either way. So it is evaluated, not trained.

## Turning it on

```console
user@host:~/lead$ python -m lead.training.train \
    training.data.use_sensor_degradation=true \
    training.data.deployment_perturbation_families='[occlusion,ego_state]'
```

| key                                                  |    default | what it does                                          |
| :--------------------------------------------------- | ---------: | :---------------------------------------------------- |
| `training.data.deployment_perturbation_families`      |       `()` | batch-level families sampled beside the appearance ones |
| `training.data.latency_curriculum_probability`        |      `0.0` | chance a sample's planning label is re-anchored        |
| `policy.transfuser.future_ego_pose_extra_ticks`       |        `0` | extra future ticks read, so a shift has somewhere to go |

Latency needs both of the last two. `future_ego_pose_extra_ticks` must be a whole
number of planning strides — a shift landing between two label ticks has no pose
to re-anchor on, and the config refuses it rather than rounding into a
mislabelled horizon.

## Two invariants worth knowing about

**The default path is unchanged draw for draw.** With
`deployment_perturbation_families` empty the sampler takes exactly the random
draws it always took. That matters more than it looks: the rung that established
the curriculum's effect is defined by this sampler's random stream, so an extra
draw on the default path would silently make it a different experiment. There is
a test.

**Reading further ahead does not widen what the model predicts.**
`num_ego_pose_prediction` is derived from the plan horizon rather than the read
window, so turning the latency curriculum on does not change the head's output
width and orphan every existing checkpoint. There is a test for that too.

The extra ticks default to zero for a related reason: every extra tick asked for
drops the scenes near the end of a log, so a rung trained with them sees a
slightly different dataset from one trained without.

## Evaluating them

New families are ordinary `--conditions modality:severity` tokens, applying the
same damage through the same functions the curriculum uses, and through the
route-seeded generator the paired protocol depends on:

```console
user@host:~/lead$ python scripts/common/run_evaluation.py \
    --models rung2a=outputs/rung2a_curriculum_only_post \
    --conditions none:0 occlusion:0.5 occlusion:1.0 ego_state:1.0 both:1.0
```

`both` damages the camera and the LiDAR together. It is not a training family —
the curriculum degrades one modality per sample by design, because the gate it
was built for needs somewhere to shift reliance to — and it is deliberately
outside the training distribution. It exists because it is the one regime
redundancy cannot cover, which makes it the regime the caution governor is built
for; see [the caution governor](caution_governor.md).

## What has not been measured

Nothing here has been through closed-loop evaluation yet. The families are
implemented, unit-tested under both runtime-type-checking regimes, and wired
into training and evaluation through the same code path — but no rung has been
trained with them and no driving score has been attributed to them. Treat the
list above as capability, not as result.
