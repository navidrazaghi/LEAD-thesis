# Observability supervision

An auxiliary head that predicts, per BEV cell and per modality, how well that
modality resolves whatever is there.

## Why

The expert decides which actors to react to by counting each one's LiDAR
returns and visible camera pixels and comparing both against a per-class
threshold (`ExpertOcclusionConfig`). That check keeps the expert from
demonstrating reactions to things a sensor-based student could never see — it
is how the visibility half of the learner-expert gap is closed on the data side.

Nothing closes it on the model side. The student's fusion treats every region of
every modality as equally informative, always, and so has no way to represent
that a region is occluded, out of frame, or beyond LiDAR range. This head is the
expert's occlusion check restated as something the student itself predicts.

## The targets come from the dataset as it stands

No new data collection. Every box already carries both counts, per tick:

| field                       | what it is                          | where it enters                      |
| :-------------------------- | :---------------------------------- | :----------------------------------- |
| `num_points`                | LiDAR returns on the box            | native field of the box-detection stream |
| `visible_pixels`            | camera-visible pixels of the actor  | driving-meta box attributes          |

`carla_decoding.box_detections_to_carla_ego_frame` merges both into the box
dicts the label builders read, so `observability.py` only has to rasterize them.

Each count becomes `min(1, count / threshold)` at the expert's own per-class
bar, so the target reaches 1 exactly where the expert would have called the
actor observed, and degrades smoothly below that. With
`observability_soft_targets=false` it is the expert's binary decision instead.

Two properties matter and are covered by tests. A count of zero is a
*measurement*, and the informative one — the actor is there and neither sensor
resolved it — so it supervises zero rather than being skipped. A count of `-1`
is an *absent* measurement, and is left unsupervised rather than being read as
an invisible actor.

Supervision is sparse: only cells a measured actor covers carry a target, and
the loss is masked to them, the same way the CenterNet heatmap only supervises
the cells it has boxes for. The targets live on the CenterNet cell grid — the
BEV raster divided by `bev_downsample_factor`, so 1 m cells with the shipped
geometry.

## Turning it on

```console
user@host:~/lead$ python -m lead.training.build_cache policy.transfuser.use_observability=true
user@host:~/lead$ python -m lead.training.train policy.transfuser.use_observability=true
```

| key                                              | default | what it does                                    |
| :----------------------------------------------- | ------: | :---------------------------------------------- |
| `policy.transfuser.use_observability`            | `false` | run the head and its loss                        |
| `policy.transfuser.observability_soft_targets`   |  `true` | graded targets rather than the binary decision   |
| `policy.transfuser.observability_head_channels`  |    `64` | feature width of the head                        |

Like every other head toggle, `use_observability` also decides whether the
targets reach the cache store, so a store built without it has to be rebuilt.
`observability_soft_targets` is in the cache fingerprint, so switching it is
caught rather than silently serving targets of the other shape.

The head reads the top-down BEV feature grid, which is now built when any of the
box, BEV-semantic or observability heads is on.

## Gating the fusion with it

The deformable operator already normalizes each query's sampled points over the
`(modality, point)` axes with one softmax, so which modality a query reads from
is decided by those logits and shifting the balance needs no new mechanism —
only a per-token, per-modality bias added before the softmax. That is the gate.

It is a separate, cheaper predictor from the dense head: one linear per fusion
block, reading the tokens that block already has. Both are supervised from the
same targets. The dense head predicts on the 1 m cell grid and is what you plot;
the gate predicts on the two fusion token grids and is what acts.

Mapping the targets onto those grids reuses the calibration. A BEV token pools
the cells it covers, averaging over the supervised ones alone rather than over
the whole block. An image token has no BEV footprint, so it takes the cell its
ray reaches — the same `image_tokens_in_bev` correspondence the reference points
use. Tokens whose ray never meets the ground stay unsupervised, which with the
shipped rig is the top half of the image.

The gate's projection is zero-initialized, so a gated model starts bit-for-bit
where the ungated one does and every difference is learned.

```console
user@host:~/lead$ python -m lead.training.train \
      policy.transfuser.backbone_target=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone \
      policy.transfuser.use_observability=true \
      policy.transfuser.use_observability_gate=true \
      training.data.use_sensor_degradation=true
```

| key                                                   | default | what it does                                |
| :---------------------------------------------------- | ------: | :------------------------------------------ |
| `policy.transfuser.use_observability_gate`            | `false` | bias the fusion's modality logits            |
| `policy.transfuser.observability_gate_loss_weight`    |   `1.0` | how hard the gate is pulled to the targets   |
| `training.data.use_sensor_degradation`                | `false` | degrade a modality per sample while training |
| `training.data.sensor_degradation_probability`        |   `0.5` | chance a sample is degraded                  |
| `training.data.sensor_degradation_max_severity`       |   `1.0` | upper bound of the sampled severity          |

The gate needs a backbone with a modality axis to shift; asking for it on the
base fusion raises at construction rather than silently doing nothing.

## Why the degradation curriculum is not optional

The dataset is recorded under one set of conditions. Nothing in it shows what a
failing camera looks like, so an observability head trained on it alone learns
occlusion only — which actor is behind which — and a gate trained on that has no
reason to react when a sensor itself degrades. The headline robustness claim
would have nothing behind it.

`sensor_degradation.py` supplies the missing variation: pick one modality per
sample, damage it by a sampled severity, and scale *that modality's*
observability targets by what survives. The pairing is the whole point. Damaging
the input alone trains the head to insist the camera still sees everything;
scaling the target alone trains it to cry wolf. Only one modality is damaged per
sample, because a gate whose job is to shift reading onto an intact modality
needs one to shift to.

Scaling the target linearly in severity is a modelling assumption, not a
measurement: the dataset records visibility under nominal sensors only, so the
degraded value cannot be observed, only posited. Real weather stays a held-out
test, which is what makes it an out-of-distribution result rather than a
memorized one.
