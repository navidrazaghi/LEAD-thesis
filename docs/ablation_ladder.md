# The ablation ladder

Seven models. Each rung changes **exactly one thing** against the rung above it,
so the difference between two adjacent rows is the value of that one change and
nothing else. A ladder built any other way produces numbers nobody can attribute,
which is worse than no numbers at all — it invites a confident claim that the
first hard question destroys.

## The pieces being varied

**Fusion operator.** The stock TransFuser fusion runs dense self-attention over
the concatenation of the image anchor grid and the BEV anchor grid: each of the
552 tokens scores against all 552, quadratic in the token count. The deformable
alternative is sparse — each query predicts a few sampling offsets per modality
and reads bilinearly interpolated values only there.

**Reference points.** When an image token samples the BEV grid, where does it
start? Deformable DETR's levels are one scene at several scales, so a normalized
coordinate carries across them unchanged. Here the two levels are different
modalities — a perspective grid and a top-down one — and no such identity holds.
Geometry-free, a query starts at the centre of the other grid and must learn the
correspondence. Calibrated, it starts where the rig says its own ray meets the
ground, and only refines from there.

**Observability head.** A decoder over the BEV feature grid predicting, per cell
and per modality, how well that modality resolves whatever occupies the cell.
The labels come from the expert: while collecting data it counts each actor's
LiDAR returns and visible camera pixels, and reacts to the actor only once both
clear a per-class threshold. That check runs on the privileged side only; this
head is its twin on the student side. Supervision is sparse — only cells a
measured actor covers carry a target — so the loss is masked.

**The gate.** The deformable operator already normalizes each query's sampled
points over the `(modality, point)` axes with one softmax, so which modality a
query reads from is decided by those logits. The gate predicts a per-token,
per-modality bias added before that softmax. No new mechanism is needed — only a
shift.

Note what follows from softmax being shift-invariant: adding the same bias to
both modalities cancels. The gate can therefore only act where the two modalities
*differ* in observability, which is exactly the intended behaviour.

**Degradation curriculum.** The dataset is recorded under one set of conditions,
so nothing in it teaches a model what a failing sensor looks like. This damages
one modality per sample by a sampled severity and scales that modality's
observability targets to match. The pairing is the point: damaging the input
alone would train the head to insist the camera still sees everything; scaling
the target alone would train it to cry wolf.

## The ladder

| rung | fusion | reference | obs head | gate | curriculum |
| :--- | :----- | :-------- | :------: | :--: | :--------: |
| rung0 | dense | — | no | no | no |
| rung1 | deformable | geometry-free | no | no | no |
| rung2 | deformable | calibrated | no | no | no |
| rung2a | deformable | calibrated | no | no | **yes** |
| rung2b | deformable | calibrated | **yes** | no | yes |
| rung3 | deformable | calibrated | yes | **yes** | yes |
| rung4 | as rung3, with both auxiliary loss weights at 0.2 instead of 1.0 | | | | |

Every rung is two runs: a pretrain (perception only) and a posttrain that starts
from the pretrain's weights and adds the planning decoder. Evaluation always uses
the `_post` checkpoint — the pretrain has no planning decoder and therefore no
waypoints.

## What each rung is for

**rung0 — baseline.** The repository as published, untouched. The reference
point every other number is read against.

**rung1 — the operator.** Swaps dense attention for deformable, with
geometry-free reference points. Isolates what the sparse operator costs or buys
on its own, before any geometric prior.

**rung2 — the calibration prior.** Same operator, reference points seeded from
the rig. Against rung1, this is what the geometry is worth.

**rung2a — the curriculum alone.** Adds degraded-sensor training and nothing
else. This rung exists because the dullest explanation for any robustness result
is that a model trained on damaged sensors drives better with damaged sensors —
gate or no gate. Without this row, no claim about the gate is attributable.

An earlier version of the chain omitted it deliberately, reasoning that "the
curriculum without the gate is just augmentation." That reasoning is backwards.
Being just augmentation is precisely why it has to be run.

**rung2b — the head without the steering.** Adds the observability head, which
is trained and whose features sit in the encoder, but nothing consumes its
output to steer fusion. Against rung2b, rung3 is the gate and nothing else.

**rung3 — the full method.** The head, the gate that reads it, and the
curriculum that gives the gate something to learn from.

**rung4 — the dilution control.** `train.py` normalizes the per-task loss
weights by their sum, so rung3 at full weight spends 2 of 12 parts on
observability and hands its driving losses 1/12 where rung2a gets 1/10. At 0.2
each the sum is 10.4 and the shares are within a few percent of each other, with
the gate still supervised. This asks whether rung3's deficit was arithmetic
rather than conceptual.

## Reproducing each rung

All of them run through `scripts/common/run_ablation_chain.sh`, which skips any
rung that already has its final checkpoint. The overrides that define them:

```bash
DEFORMABLE=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

# rung0  (no overrides beyond the posttrain wiring)

# rung1
policy.transfuser.backbone_target=$DEFORMABLE

# rung2
policy.transfuser.backbone_target=$DEFORMABLE
policy.transfuser.deformable_calibrated_reference=true

# rung2a
policy.transfuser.backbone_target=$DEFORMABLE
policy.transfuser.deformable_calibrated_reference=true
training.data.use_sensor_degradation=true

# rung2b
policy.transfuser.backbone_target=$DEFORMABLE
policy.transfuser.deformable_calibrated_reference=true
policy.transfuser.use_observability=true
training.data.use_sensor_degradation=true

# rung3
policy.transfuser.backbone_target=$DEFORMABLE
policy.transfuser.deformable_calibrated_reference=true
policy.transfuser.use_observability=true
policy.transfuser.use_observability_gate=true
training.data.use_sensor_degradation=true

# rung4
# ... as rung3, plus:
policy.transfuser.observability_loss_weight=0.2
policy.transfuser.observability_gate_loss_weight=0.2
```

`use_observability` also decides whether the targets reach the cache store, so
turning it on needs a store built with it on.

## What has been measured so far

**Open-loop only.** Everything below is waypoint L2 and attention statistics over
dataset frames, from `scripts/common/analyze_gate.py`. Open-loop error averages
over frames and easy frames dominate it, while driving fails in rare moments, so
these numbers **triage** — they decide which models are worth an expensive
closed-loop budget. They do not settle the claim.

The mechanism works. Shift in the camera's share of attention mass, relative to
each model's own clean row:

| rung | gate | mean shift | correct direction |
| :--- | :--- | ---------: | :---------------: |
| rung1 | no | 0.0056 | 2/4 |
| rung2 | no | 0.0035 | 2/4 |
| rung2a | no | 0.0127 | 4/4 |
| rung2b | no | 0.0044 | 4/4 |
| rung3 | **yes** | **0.1700** | **4/4** |

The ungated models are deformable too, so the operator is not what reallocates —
they simply have no modality axis to shift. The ungated shifts also land on the
correct side only half the time, which marks them as noise rather than a weak
version of the same response.

The performance did not follow. Waypoint L2, lower better:

| rung | clean | camera:1.0 | lidar:1.0 |
| :--- | ----: | ---------: | --------: |
| rung0 | 0.691 | 0.818 | 1.288 |
| rung1 | 0.492 | 0.675 | 1.005 |
| rung2 | 0.555 | 1.093 | 1.643 |
| **rung2a** | **0.442** | **0.568** | 0.689 |
| rung2b | 0.557 | 0.694 | **0.667** |
| rung3 | 0.581 | 0.635 | 0.800 |

rung2a — curriculum only, no gate — is best in four of five columns, clean
included. rung3 wins none.

Beware the percentage view of this table. Expressed as increase over each
model's own clean row, rung3 looks best by a wide margin, but only because it
starts from a worse clean score and therefore has less room to fall. The
absolute table is the one that decides.

Two things follow that are worth carrying into the write-up. The curriculum is a
strong regularizer — rung2a is the best model overall, not merely the most
robust one — and the clean-performance cost in rung3 traces to the auxiliary
tasks rather than to the curriculum: 0.442 → 0.557 on adding the head, → 0.581
on adding the gate.

## What is still open

The closed-loop campaign: rung0, rung2a, and the better of rung3/rung4, over 30
stratified Bench2Drive routes at three conditions. Route count is set by
measured spread — the per-route driving score standard deviation is about 31, so
ten routes could only resolve a 12–20 point difference against a mean near 30.
Thirty routes brings that to roughly 7–11 points.
