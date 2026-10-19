# Transformer agent engineering check — October 19, 2026

At Clark's request, the current agent uses a causal transformer in place of GRU
memory, with a readable model file and high-level component map in the model README.
This check verifies implementation and saved weights, not survival improvement.

- Model: one layer, four heads, width32, context16; four categorical variables
  with four categories; linear actor/value heads.33,601 parameters including the
  frozen target-value head. No pretrained language model.
- World fitting:999 real active-agent transitions from two canonical JavaScript
  synthetic-v2 worlds; eight optimizer updates. Attention weights changed.
- Policy fitting:two updates,32 joint steps and248 active-agent transitions.
  Actor weights changed; world-model weights remained exactly frozen.
- Reload: safe checkpoint reproduced every sampled frame in replay-check
  world1020000. This world is an engineering check, not a final comparison suite.
- Checkpoint SHA256:
  `3a0257b8691b48c843615272e3fdd85ff2abd161080fe1a8ce681f6828dab1de`.
- CPU wall time1.51 seconds, excluding test suites; paid cost$0.
- Tests:41 model tests and71 integration tests passed, including causal attention,
  per-agent reset/memory isolation, real attention gradients, independent environment
  alignment, burn-in gradient boundaries and exact optimizer-resume regression.

Private artifacts are in the existing project's `.private/transformer-agent-smoke-001/`:
`protocol.json`, `report.json`, `transformer-smoke.safe`. Weights are not published.
This integration checkpoint saves weights, not a full mid-episode training resume.
The protocol was written before fitting; no final-world selection was performed.

No new held-out performance claim, retention result or continual-learning mechanism
follows from this check. The earlier GRU reports and checkpoint bytes remain intact.
The website remains scripted. Larger training/comparison requires a separately
bounded protocol and fresh held-out worlds. No training remains running.
