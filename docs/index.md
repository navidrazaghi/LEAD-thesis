# LEAD

LEAD produces an as-generic-as-possible driving dataset, plus the generic tooling
around it.

Most CARLA codebases are written around one model; LEAD is written around the data,
in the [Py123D](https://github.com/kesai-labs/py123d) standard format.

Reading the data is model-agnostic, while featurization and label building are
entirely the policy's responsibility. This split is enforced by import contracts,
not by convention.

## Basic documents

To understand the data layout and how the data is processed:

- [Data access](data_access.md): understand the data layout.

To understand how TransFuser is trained, and how to implement your own policy:

- [Training](training.md): building the cache, pretraining, post-training, config overrides.
- [Add your own policy](add_a_policy.md): shortest path to add your own policy.

To generate your own data, see:

- [Data collection](data_generation.md): running the expert to generate a dataset, changing the sensor rig, adding a modality offline.

## Further documents

Our repository follows tested software-engineering practice, with one deliberate
bias: where clean abstraction and data-loading throughput conflict, we trade a
little abstraction for throughput.

- [Architecture](architecture.md): understand the design of the repository and how data is loaded and processed.

## Observability-gated fusion

The work this fork adds: a deformable fusion operator whose modality weights are
shifted by a predicted per-cell observability, trained against sensor
degradation. Read them in this order.

- [The ablation ladder](ablation_ladder.md): the seven models, what each one
  isolates, how to reproduce them, and what has been measured. **Start here** —
  the rest only makes sense against the ladder.
- [Deformable fusion](deformable_fusion.md): the sparse attention operator that
  replaces the dense fusion transformer, and the calibrated reference points.
- [Observability](observability.md): the per-cell, per-modality supervision, the
  gate that consumes it, and the degradation curriculum.
- [Running on the lab A100](server_setup_notes.md): environment facts and the
  traps that cost real time — several of them fail quietly rather than loudly.

## Robustness beyond the appearance families

Work in progress on the `robust-deployment` branch, built against what the
ladder measured rather than alongside it. One of the two has an answer already,
and it is negative.

- [Deployment perturbations](deployment_perturbations.md): occlusion, ego-state
  noise, execution latency and frame freeze — why latency is implemented as a
  planning-label re-anchor rather than a delayed observation, and the joint
  degradation condition. Not yet evaluated; a rung carrying the first two
  families is in training.
- [The caution governor](caution_governor.md): the same observability signal
  actuated in the controller instead of the attention logits, three ways to
  measure caution, and an online calibrator in place of a tuned threshold.
  **Result: no leverage in this stack.** The mechanism works as specified and
  all three signals are swept across seven conditions, but no condition gives it
  both a signal and something to gain: where the fault is real nothing drives —
  the published reference completes 3% of a route with both sensors destroyed —
  and where the fault is detectable the policy already copes. The document
  carries the runs and the sweep, including a correction to a claim it made
  earlier from a cheaper proxy that was an order of magnitude out.
