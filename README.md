# Survival Agent Pilot

[![Actual recorded survival simulation: scripted greedy baseline, seed 0; not trained AI](assets/survival-preview.gif)](https://clarkwhitehead.com/)

**Actual recorded run · scripted greedy baseline · no trained result.** Click the preview for the replay controls on the website.

A small, inspectable multiagent survival research scaffold. **This is an engineering prototype, not evidence of learned survival, emergent cooperation or continual learning.**

Start reading:

- [MuZero-style model, PUCT MCTS, replay and recurrent losses](survival_agent/muzero.py)
- [Small transformer-history agent](survival_agent/agent.py)
- [Local pretrained/scratch language backend](survival_agent/language_backend.py)
- [Language-feature-to-latent controller](survival_agent/integrated.py)
- [Actor-critic baseline trainer/evaluator](survival_agent/training.py)
- [Simulator and observation-limited baselines](survival_agent/simulator.py)
- [Security scope and limitations](SECURITY.md)

## Layout

```text
survival_agent/   Models, simulator, training and CLI modules
tests/            Correctness and security regression tests
scripts/          Explicit model download and preview utilities
docs/             Architecture and usage notes
assets/           Small recorded preview and provenance
```

## Environment

16×16 grid; eight initially equal agents; move/wait actions; food, hunger and death; 5×5 local observations. Simultaneous contested moves reject all contenders. Reward is one per agent alive after a transition, zero otherwise. Seeded bounded episodes and private observation histories are implemented. Optional plain-text communication is range-limited; no cooperation reward is added.

## Distinct learning paths

The **actor-critic baseline** uses fresh on-policy survival data and auxiliary observed-transition reward/dynamics targets. The separate **minimal MuZero-style option** uses actual PUCT visit-frequency policy targets, episode replay, n-step values and recurrent unroll losses. It is not a full MuZero reproduction: deterministic latent transitions approximate a partially observed world with interacting agents. No validated planning advantage is claimed.

Scratch MuZero training updates its small observation transformer, dynamics and heads. The optional language-feature route uses a frozen local encoder and trainable projection/latent heads. Pretrained encoder fine-tuning is explicitly unsupported. The separate fully scratch language configuration uses random weights and a fixed UTF-8 byte vocabulary; natural-language competence is not claimed.

Pretrained language defaults to Qwen/Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`. Weights are not included. Loading is local-directory-only, safetensors-only and `trust_remote_code=False`. The optional `scripts/fetch_qwen.py` helper downloads only that pinned official artifact when explicitly invoked. The model's license is Apache-2.0; no project license has been selected.

## Local setup

Use a fresh virtual environment with Python 3.10+ and the security-oriented dependency minimums in `requirements-language.txt`. Do not reuse an unpatched ML environment for untrusted files. See SECURITY.md for advisory review and compatibility limits.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-language.txt
python -B -m unittest discover -v
```

No package installation is performed by project scripts. The current publication was functionally checked on the pre-existing legacy runtime; the newly recommended dependency combination has not been fully retested. This is not a security certification of that legacy runtime.

## Bounded examples

Use new output directories. These commands are small engineering checks, not scientific training runs:

```bash
# Recorded scripted baseline and offline replay.html
python -B -m survival_agent.run --policy greedy --seed 0 --steps 128 --output runs/baseline

# Minimal MuZero-style scratch collection/replay update
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.muzero_run --output runs/muzero --steps 4 --simulations 8 --updates 1 --max-seconds 8

# Actor-critic collection/training/evaluation remain separate
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.training collect --output runs/data --max-joint-steps 2
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.training train --output runs/ac --max-joint-steps 2
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.training evaluate --checkpoint runs/ac/checkpoint.safe --output runs/eval --eval-seeds 1001 --horizon 2 --max-joint-steps 8
```

Checkpoint/resume uses schema-validated JSON and safetensors in a bounded archive, **not pickle or torch.load**. Legacy checkpoint formats are rejected. The MuZero learner supports exact next replay-update resume at episode boundaries; the actor-critic checkpoint also captures environment/history/sampler state. Runtime limits are cooperative at step boundaries; shell `timeout` gives a hard process cap.

## Limits

No useful learned policy, language cooperation, humanlike civilization, generalization, or cross-world retention is established. Speech learning, independent-agent optimization, large populations, pretrained fine-tuning, full MuZero scaling/reanalysis and meaningful scientific evaluation remain future work. No GPU, API service or network server is started automatically.

This repository contains curated source and a recorded preview with provenance. Model weights, checkpoints, raw run datasets, personal context and earlier private development history are excluded. Existing local work/history was preserved separately; this is a clean public source history rather than a rewritten remote history.
