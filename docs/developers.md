# Developer guide

The first package split is implemented: `survival_env` contains mechanics, `survival_agent` contains models/search, and `survival_adapters` contains integration. Existing CLI/import paths forward to these implementations. All commands below run from the repository root on a POSIX shell with Python 3.10+; `timeout` is the GNU utility. Windows users can use WSL. Output folders must be new.

- [Current interfaces and extension examples](interfaces.md)
- [Models, checkpoints and provenance](models.md)
- [Tests and command verification](verification.md)
- [Directory and architecture map](architecture.md)
- [Separation status](separation-plan.md)
- [Security scope](../SECURITY.md)

## Setup status

The fresh dependency combination recommended in [requirements-language.txt](../requirements-language.txt) is **not yet compatibility-tested**. The available legacy runtime passes the functional checks in [verification](verification.md), but has known advisories. Do not describe it as a supported secure installation.

The setup commands below are instructions for an isolated environment, not a completed clean-install test. Installing ML dependencies requires network access and potentially substantial disk space. No dependencies are installed by our verification script.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-language.txt
```

For environment-only use, `requirements.txt` and the base package metadata require only NumPy. `requirements-agent.txt` adds Torch/safetensors; `requirements-language.txt` adds Transformers/Jinja2. Package extras `agent` and `language` expose the same distinction. There is one distribution containing three importable packages, not separately published distributions. No install or dependency resolution was performed for this split.

## First run

This records a scripted baseline, without model loading, training or network access:

```bash
python -B -m survival_agent.run --policy greedy --seed 0 --steps 4 --output runs/quickstart
```

Expect `summary.json`, `frames.json` and a self-contained `replay.html`. Open the HTML in a browser manually. The full-world replay is for inspection; the policy receives local observations. Rerunning with an existing output folder intentionally fails.

## Extend the current code

```bash
python -B -m examples.custom_policy
python -B -m examples.toy_env_adapter
```

The first example uses a scripted wait policy and completes three joint steps with reward 24. The second feeds a public bandit cue into the existing randomly initialized latent core. It demonstrates a narrow five-action adapter, with no language model and zero optimizer updates. Neither is a trained result.

## Test before changing behavior

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m unittest discover -v
python -B -m scripts.verify_docs --no-training
```

The second command inventories and checks every shell command in README and `docs/`, runs permitted nontraining examples in temporary output folders, verifies outputs and links, and reports skipped setup/download commands explicitly. The `--no-training` pass skips experiment collection/training/evaluation commands. The explicitly requested regression suite still exercises tiny deterministic update/resume cases; no training experiment is launched. See the [verification limits](verification.md).
