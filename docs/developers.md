# Developer guide

Start with the working code before the planned package split. All commands below run from the repository root on a POSIX shell with Python 3.10+; `timeout` is the GNU utility. Windows users can use WSL. Output folders must be new.

- [Current interfaces and extension examples](interfaces.md)
- [Models, checkpoints and provenance](models.md)
- [Tests and command verification](verification.md)
- [Directory and architecture map](architecture.md)
- [Planned separation](separation-plan.md)
- [Security scope](../SECURITY.md)

## Setup status

The fresh dependency combination recommended in [requirements-language.txt](../requirements-language.txt) is **not yet compatibility-tested**. The available legacy runtime passes the functional checks in [verification](verification.md), but has known advisories. Do not describe it as a supported secure installation.

The setup commands below are instructions for an isolated environment, not a completed clean-install test. Installing ML dependencies requires network access and potentially substantial disk space. No dependencies are installed by our verification script.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-language.txt
```

For environment-only use, the existing simulator imports NumPy without Torch or Transformers. Packaging still lists ML dependencies; independent installable packages are planned. Do not infer that an environment-only distribution already exists.

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
python -B -m scripts.verify_docs
```

The second command inventories and checks every shell command in README and `docs/`, runs permitted bounded examples in temporary output folders, verifies outputs and links, and reports skipped setup/download commands explicitly. It runs small scratch-model engineering updates, not pretrained inference or a study. See the [verification limits](verification.md).
