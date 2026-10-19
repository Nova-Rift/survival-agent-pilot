# Why synthetic-control-001 scored below random

October 19, 2026. A frozen-checkpoint audit found a concrete train/evaluation
mismatch: training sampled categorical actions, while the original score used
argmax actions. This measured a different controller. Neither evaluation mode is
inherently invalid, but the difference was undeclared and confounded the result.
The original argmax scores are preserved and remain valid for that controller.

## Controlled check, without retraining

The original selected checkpoint was held fixed (SHA256
`4fcaa44a9276c372d8d057855a2d7fa59f2a11693c8a7370069d90535ddfa85d`). No optimizer
steps, candidate reselection or reward/environment changes occurred. The protocol
specified eight fresh diagnostic world seeds450000–450007 and sixteen fresh final
world seeds660000–660015 before outcomes. Original001 final seeds330000–330015 were
not reused. Six controller variants were fixed in advance, including matched
untrained/sampled references with identical action RNG seeds. Each world is the
comparison unit; its eight competing agents are not independent replications.

| Controller | Diagnostic mean | Fresh final mean |
|---|---:|---:|
| Trained, argmax | 26.125 | 25.9609375 |
| Trained, categorical sampling | 59.828125 | 60.2265625 |
| Untrained, argmax | 25.953125 | 27.125 |
| Untrained, categorical sampling | 57.421875 | 64.5 |
| Uniform random | 67.890625 | 61.1015625 |
| Local scripted | 81.625 | 81.6328125 |

Scores are mean per-agent alive-step returns, ceiling95 under unchanged wear96
and zero reward on the death transition. Trained sampling lost to untrained
sampling in11/16 paired final worlds (mean difference−4.2734375). Its mean was
0.875 below random. These results establish no useful learned-control improvement.

## Concrete behavior and optimization findings

Under argmax, the trained checkpoint chose action4 (left) on100% of active-agent
decisions in the fresh final worlds. Of those move attempts,71.83% left position
unchanged. It consumed1.05 resources per agent. Its categorical distribution was
still broad: entropy about1.60 nats (uniform maximum ln5≈1.609), average largest
action probability only0.245. Taking its small score preference deterministically
caused repeated leftward movement and stalls, rather than its trained exploration.

Sampling spread its actions approximately16.93/21.44/20.71/17.24/23.68% across
wait/up/right/down/left. Stationary move attempts fell to9.54%, resources rose to
7.93 per agent, and survival approached random. This explains much of the
argmax-versus-random gap; it does not demonstrate learning. Untrained sampling
remains stronger on these worlds, and the scripted reference is substantially
better. A stationary move can result from boundaries or occupied/contested cells;
the audit does not assign every stall to a particular cause.

The mode-latent posterior used only one categorical combination per evaluated
world. Continuous GRU features still varied (nonzero feature variance), so this
is not proof the entire representation is constant or incapable of control.
It is evidence against claiming rich discrete learned state from this checkpoint.

All original optimization losses/gradients were finite. Mean entropy stayed
near-uniform (first8 updates1.6001; last8 1.5870). Mean critic MSE remained high
(76.01→62.48). A16-step real rollout inspected at the saved checkpoint had mean
value0.277 versus n-step target7.780. The half-weighted critic gradient norm was
19.57 versus actor0.150; combined clipping at1 strongly scales the actor gradient
with the critic. This is a concrete optimization limitation and a possible future
controlled probe, not an established causal bug. No clipping/loss coefficient,
model, reward or compute-budget change was made in response to final scores.

## Correctness audit and fix

- Actual200-dimensional observations are8×5×5 local channels from the canonical
  JavaScript simulation. Training's features, rewards, actions and terminal flags
  match an independent same-action replay exactly; next state uses the actual
  previous action and next observation. No overhead frame reaches the actor.
- Reward remains1 after an alive transition and0 on death; no reward shaping.
  Inactive targets are zero, true death blocks bootstrap, and a truncation uses
  its same-episode final value then cuts reset recursion. Default wear96 occurs
  before horizon128, so truncation semantics also have a targeted artificial test.
- Training and evaluation use the same posterior mode features and reset routine.
  Separate bridge processes isolate validation from training-world state.
  Checkpoint bytes and all model parameters stayed unchanged throughout the audit.
- A shared `policy_actions` helper now defines sampled versus greedy semantics.
  New training runs explicitly evaluate/sample the categorical policy with a fixed
  independent RNG; sampled recording reload is deterministic. The legacy
  `evaluate()` greedy default remains for explicit reproduction of001, and returned
  results always identify action mode. No new training run was necessary to
  isolate this evaluation defect.

Seven focused tests pass, covering RNG sampling versus argmax, observation/action
alignment, terminal/inactive masks, truncation/reset bootstrap, frozen weights,
evaluation isolation and reproducible sampling without global RNG mutation.
All68 integration tests pass; unchanged model/JS suites previously passed34/35.

## Evidence and verification limits

- [Frozen protocol](synthetic-control-diagnosis-001/protocol.json)
- [Full per-world telemetry, scores and optimization audit](synthetic-control-diagnosis-001/report.json)
- [Original negative experiment, unchanged](synthetic-control-001.md)
- Reproduce with `examples/diagnose_synthetic_control.py`, the original trusted
  local checkpoint, canonical environment directory, and a fresh output folder.

The audit took about12 seconds on CPU and cost$0 paid compute. No retraining, GPU,
new package, CBP or architecture redesign was used. Supported browser inventory
was freshly checked and returned `apps:[]`, `browsers:[]`; version19 mobile layout
and hydration remain unverified. No screenshot was taken. The published homepage
stays honestly labeled scripted. Stop this pass here; preserve these results
before any separately bounded representation/optimization probe on new seeds.
