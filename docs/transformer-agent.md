# Transformer agent: integration and verification

Clark requested transformer memory and a simple, inspectable implementation on
October 19, 2026. The active model and action selection live in
[`transformer_agent.py`](https://github.com/Nova-Rift/survival-agent/blob/hybrid-stage-b/survival_agent/hybrid/transformer_agent.py).
The model repository's [README](https://github.com/Nova-Rift/survival-agent/blob/hybrid-stage-b/README.md)
explains each component, including what is still unimplemented.

## Run the current path

From this checkout, using the existing sibling source trees and installed dependencies:

```bash
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B -m examples.train_synthetic_control \
  --environment ../survival-env/browser \
  --output /absolute/private/path/transformer-smoke-new
```

The output directory must be new. This command now selects the transformer and
runs a deliberately small **engineering check**: two prefit worlds, eight
world-model updates, then two actor/critic updates (32 joint simulation steps).
It saves weights and verifies identical sampled rollouts after checkpoint reload.
It does not run a held-out performance comparison or select a checkpoint by score.
The historical GRU runner remains `survival_adapters.synthetic_control.run`;
its prior final suites must not be reused for tuning a new architecture.

The environment is the canonical JavaScript synthetic-v2 implementation, accessed
through the existing local Node bridge. There is no Python rules fork. The
transformer has one layer, four heads, width32 and a16-token attention window.
It processes previous categorical state/action tokens; the current observation
enters the posterior. Memory is per agent and cleared independently at resets.
Attention is causal, right-padding is masked, and the oldest raw token is evicted
when the window is full. This is a small standard pre-layernorm PyTorch transformer,
not a language model or a claim to reproduce GTrXL.

## Where to read the rest

| File | Responsibility |
|---|---|
| Model repository `hybrid/transformer_agent.py` | All current model layers, history updates, prior/posterior, predictions, actor, critic and action sampling. |
| Model repository `hybrid/world_model.py`: `sequence_loss`, `WorldModelTrainer` | Shared replay-sequence losses and world-model optimization; historical GRU class remains there. |
| `survival_adapters/transformer_control.py` | Current run budgets, model construction, real-data collection, safe checkpoint loading and replay check. |
| `survival_adapters/synthetic_control.py`: `SyntheticTrainer` | Existing real-rollout actor/critic training reused by the transformer. |
| `survival_adapters/training.py`: `survival_targets` | Returns, terminal/truncation handling and inactive-agent masking. |
| `scripts/synthetic-bridge.mjs` | Calls the exact browser environment; model input stays local. |

No alternate learning algorithm is hidden inside the transformer. The actor/value
losses, survival reward and frozen-world training phases match the existing path.
Continual-learning replay schemes, neuron recycling, old-skill retention tests,
language input and learned planning are not implemented by this change.

## Verification and checkpoint meaning

Tests cover causal attention (training and evaluation), no current-observation
leak into the prior, sliding memory, per-stream resets, finite attention gradients,
fixed-fixture fitting, burn-in gradient boundaries, frozen world weights during
policy updates, exact next-update checkpoint resumption, and observation/action
alignment against independent replay of the real environment.

The integration checkpoint is explicitly tagged `transformer-posterior-control-v1`.
Loading verifies source hashes, specification and architecture/configuration before
weights. GRU checkpoints are rejected, not relabeled or converted. Integration
artifacts save selected weights, not a mid-episode optimizer/environment resume;
the generic model checkpoint regression separately verifies optimizer resumption.

The smoke seeds are training1010000–1010001, policy training beginning150000, and
replay check1020000. None is a reused final evaluation suite. No survival superiority
can be inferred from this smoke. A future performance run needs a bounded protocol
with fresh held-out worlds and reference controls. Public demo status remains scripted.
