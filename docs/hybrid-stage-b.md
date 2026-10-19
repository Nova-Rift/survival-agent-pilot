# Stage-B survival integration

This branch (`hybrid-stage-b-survival`) owns only the concrete survival adapter,
real replay collection and smoke example. The RSSM and trainer are independently
maintained in sibling `survival-agent` branch `hybrid-stage-b`; environment rules
remain in `survival-env`. No model/environment implementation is copied here.
The installed/public dependency pins still point at older extraction commits;
use the following explicit sibling-checkout workflow, not a claimed fresh install:

```sh
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m unittest discover -v
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m examples.hybrid_stage_b --output runs/hybrid-stage-b --updates 40
```

Choose a new output directory. The first command passed **56 tests**, including
four new integration tests. The second collects 16 complete training episodes
from random/local-greedy policies and four separate held-out episode seeds. It
performs 40 world-model updates on CPU, batch four, burn-in one, unroll three;
at most 100 updates are accepted. No packages, pretrained weights or services are
downloaded. No GPU or paid compute is used. The pre-existing legacy runtime's
security limitations remain; this is not a fresh-install/security certification.

## Adapter semantics

`survival_adapters/hybrid_survival.py` exposes one agent's copied local observation
as a vector. Default is a six-by-six grid, radius one (45 vector features), hunger
four, three initial food cells, no food respawn and horizon eight. No water,
aging, injury or other mechanics were added. The adapter retains death and time
limit as separate flags, including when both occur on the same transition.
Explicit reset constructs the next world; the final observation is stored first.
Dead final observations are zero under existing environment rules, while a
surviving horizon-truncated agent retains its local observation. Model code does
not import this adapter or inspect hidden world state.

The collection is scripted; no learned policy acts in the world. Reward remains
one for alive after the transition. The model predicts continuation
`1 - terminated`, not `1 - (terminated or truncated)`. Tests check exclusive
truncation as well as death and prevent reset observations entering old episodes.
The particular recorded seed-0 collection ended in death in every episode, with
two training and one held-out episode also reaching the horizon. Thus exclusive
truncation behavior has unit-test coverage, not data coverage in this smoke run.

## Recorded smoke

[Structured report](experiments/hybrid-stage-b-seed0/report.json) ·
[All 40 update logs](experiments/hybrid-stage-b-seed0/updates.jsonl).
The report binds model source/config/runtime, actual counts and seed values.
It includes no model weights, private paths, credentials or environment state
from unrelated work. Runtime checkpoints remain outside the repository.

The run collected **79 training transitions** and **20 held-out transitions**.
Fixed evaluation rows are sampled with replacement from the separate episode
sets. The 18,639-parameter model's mode-latent training loss changed from 2.03367
to 1.40139; held-out loss changed from 2.04366 to 1.55207. One/three-step prior
observation MSE improved, but three-step reward MSE worsened from .14791 to
.58644 and three-step continuation BCE worsened from .67527 to 1.46046.
The held-out prefix is inferred from real observations; future predictions use
prior dynamics with recorded actions, and future observations are targets only.
Categorical-mode rollouts are approximations, not stochastic-outcome calibration.

Safe structured checkpoint diagnostics matched after model/optimizer/replay/RNG
loading with a fixed caller-created architecture; no pickle loading occurs.
The core model tests additionally verify that the next stochastic optimizer
update matches exactly after restoring all those components. Checkpoint config
includes the model-source hash; source/runtime mismatch must not be bypassed.

These results verify working model fitting and diagnostics, **not learned
survival, retention, planning benefit or robust future-reward prediction**.
Actor/value learning, planners and comparisons remain pending. Investigate the
negative open-loop reward/continuation result and broader held-out data before
depending on the model for control. The live website replay is unchanged.
