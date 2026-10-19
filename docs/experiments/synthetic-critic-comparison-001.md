# Bounded critic optimization experiment — October 19, 2026

The critic was under-optimized at the original budget, but better value fitting
did not improve survival in this bounded comparison. No trained policy earned
deployment. The website stays on its honestly labeled scripted controller.
Default training retains one critic update per actor update; eight is an explicit
experimental option, not a claimed better default. Prior negative results remain.

## Diagnosis before training

The [previous audit](synthetic-control-diagnosis-001.md) verified correct death,
truncation, inactive-agent and observation/action handling. Existing survival
rewards are dimensionless0/1; value targets are discounted alive-step returns,
not probabilities. Their magnitude around8 on16-step rollouts is expected.
The original critic's mean prediction around0.3 was therefore underfitting,
not a reason to normalize rewards or manufacture a different objective.

The existing combined loss is actor +0.5 value MSE −0.01 entropy, Adam0.001,
gradient clipping1. The critic gradient dominated the actor gradient in the
previous inspection. To separate optimizer budget from feature representability,
a copied linear critic was fitted to frozen existing targets; original actor,
model, target generator and checkpoint bytes stayed untouched. Four rollouts
starting at world111000 supplied460 active examples; four independent rollouts
starting112000 supplied342 development examples. No evaluation-final worlds were
used to select this intervention.

| Copied critic updates | Fit MSE | Development MSE | Mean development value |
|---|---:|---:|---:|
| 0 | 74.64 | 72.39 | 0.319 |
| 128 | 51.98 | 50.79 | 2.062 |
| 1,024 | 19.50 | 19.96 | 7.535 |
| 4,096 | 15.29 | 14.74 | 7.987 |

Development target mean was7.644. Float64 least-squares fitting of the same linear
head reached fit14.22/development13.08, with feature/intercept rank33. This fixed
data check supports insufficient value optimization at128 updates; it finds no
new target/scale implementation defect. It does not establish that prediction
error is the cause of weak policy learning. The probe took2.03 seconds on CPU.

## One fixed intervention, then frozen evaluation

Before training, the experiment specified critic-update ratios1 versus8, three
matched policy seeds123/456/789, and128 actor updates ×16 joint steps per run.
This is768 actor updates and12,288 joint steps over six runs, with63,994 actual
active-agent transitions. Active exposure can differ because policies change
trajectories; joint-step budgets and initial states are matched.

Architecture and frozen world-model weights are identical across arms: the
existing32-unit GRU/categorical RSSM posterior with shared linear actor/value.
All arms reuse original001's fitted world model (one representation seed), with
new matched actor/value initializations. No prior planning, CBP, intrinsic reward,
new model, GPU, package or reward/environment change. CPU only, paid cost$0.

The eight-update arm adds seven critic-only Adam steps on the same frozen n-step
targets from each rollout, at the same learning rate,0.5 MSE weight and clip1.
Those extra steps do not update the actor, frozen representation or environment.
Target EMA refreshes exactly once per actor update, after the final critic fit;
default ratio1 is unchanged. Thus this tests a bounded value-optimization budget,
not a full optimizer/loss-balancing redesign.

Training worlds start120000; policy sampling uses its matched initialization seed.
Each run's updates32/64/128 are ranked only on eight development worlds700000–700007,
with earliest-update ties. All six checkpoints and their hashes were saved before
any final-world evaluation. Sixteen newly reserved final worlds800000–800015 were
then evaluated once with explicit categorical sampling and independent fixed
world-seed+11 action RNG. Untrained controls are matched per policy seed; random
and scripted controls use the same world set. Seeds from previous final suites
330000/660000 were not repurposed for tuning.

## Actual survival results

Mean per-agent alive-step return, ceiling95. Each world contains eight competing
agents; worlds, not individual agents, are the paired comparison unit.

| Policy seed | Untrained sampled | 1 critic update | 8 critic updates |
|---|---:|---:|---:|
| 123 | 63.08594 | 68.03125 | 64.87500 |
| 456 | 63.11719 | 62.24219 | 62.97656 |
| 789 | 63.53125 | 65.65625 | 64.57031 |
| Mean across policy seeds | 63.24479 | 65.30990 | 64.14063 |

Uniform random scored64.73438 and local scripted75.78906. The extra-fitting arm
lost to the baseline on two of three policy seeds and averaged1.16927 lower.
It did not improve reliably over matched untrained or random control, and every
run scored below scripted. No statistical or generalized learning claim follows
from these three policy seeds sharing one world model and one small world suite.

Selected actor updates were64/128/32 for ratio1 and32/64/64 for ratio8. No final
result changed selection or prompted another training configuration. All six
safe checkpoints reproduced their sampled fixed-seed70000 rollout after reload.
The predeclared deployment gate required reproducible improvement over scripted,
random and matched untrained for every policy seed, including paired-world wins.
Every run failed that gate; the homepage was not modified.

## Value fitting improved without corresponding control benefit

At the last eight training updates, ratio1 mean values were1.58/1.87/1.65 against
mean targets8.30/8.22/8.25; post-update MSE64.25/59.44/62.60. Ratio8 values were
10.84/10.36/10.56 against targets11.58/11.45/11.26; post-fit MSE29.59/31.01/24.55.
The target scale itself moves as the target critic learns. These online losses
are on different policy trajectories and target values, so they are not a common
fixed-data metric. The separate frozen-target probe supplies that comparison.
Improved critic fitting alone was insufficient to improve this policy reliably.

## Evidence, tests and limits

- [Preregistered comparison](synthetic-critic-comparison-001/protocol.json)
- [All six selections frozen before final evaluation](synthetic-critic-comparison-001/selection-frozen.json)
- [Full training logs and per-world results](synthetic-critic-comparison-001/report.json)
- [Fixed-target critic fit probe](synthetic-critic-comparison-001/fixed-target-probe.json)

Scripts: `examples/probe_synthetic_critic_fit.py` and
`examples/compare_synthetic_critic.py`; each accepts the original trusted private
checkpoint, canonical environment directory and a fresh output directory. Private
selected weights and sampled recordings are not published. Checkpoints contain
selected model weights, not full mid-run training resumes.

Eight focused and all69 integration tests pass. The new regression checks exactly
one actor optimizer step versus eight value steps, unchanged actor and world-model
weights during additional critic fitting, reduced fixed-batch loss, and one final
target EMA refresh. No new return-calculation defect was discovered. Existing
model/environment/site source was not changed. Paid compute remained$0.

The six-run training/development/final comparison took55.35 seconds wall time on
CPU, plus2.03 seconds for the pre-training fit probe; test time is separate.
Browser layout/hydration remains blocked by empty supported browser inventory.
Stop this experiment here. Preserve this negative intervention result and reserve
fresh final worlds for any later independently justified probe; do not extend
budgets or change objectives in response to these final scores.
