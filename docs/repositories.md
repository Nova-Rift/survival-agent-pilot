# Repository ownership and local workflow

[Developer index](developers.md)

## Current hybrid source workflow — October 19, 2026

The current model dependency is `survival-agent` commit
`4b54ed9b08ad0d4bd9e23126ed85421802b48dbb` on `hybrid-stage-b`.
The tested environment rules are commit `e4d3f66a3f870fefe7340e35f206b587eb0c69e4`
on `browser-environment`. `pyproject.toml` pins those versions. Use this integration
branch `hybrid-stage-b-survival`. The full loop now lives in the model repository:
`python3 -B -m survival_agent.hybrid.model --help`. The older
`python3 -B -m examples.hybrid_combination` integration check remains available. The
[transformer guide](transformer-agent.md) retains the posterior-only baseline command. Installed dependencies and sibling source checkouts
were tested; a new installation/wheel build remains unverified.

The extraction pins and setup narrative below are historical. They do not contain
the current transformer implementation; do not switch to them for the new command.

## Historical extraction record

| Source | Canonical repository and pinned extraction |
|---|---|
| Environment | [Nova-Rift/survival-env](https://github.com/Nova-Rift/survival-env), `3915226ceabc42e4548fc3b4b45bf146c408755b` |
| Agent | [Nova-Rift/survival-agent](https://github.com/Nova-Rift/survival-agent), `e3007112887c25b2d17a7a7c3576d53322ceccd2` |
| Integration | [Nova-Rift/survival-agent-pilot](https://github.com/Nova-Rift/survival-agent-pilot), adapters/examples/tests/demo |

The existing pilot history is retained. Environment/model source is extracted from pilot commit `be2e3f4`; new work belongs in the canonical repository. Do not maintain copied source here. Integration package metadata pins exact Git source commits rather than guessing a PyPI package. Both canonical repositories are now public. Their extraction commits and every published source file were verified before the integration update. Pins above intentionally retain the tested source commits; later publication-record commits only update verification documentation.

## Source checkout workflow, without installing anything

Clone each repository into sibling folders named `survival-env`, `survival-agent`, and `survival-agent-pilot`. The integration folder can have another name; the two dependency folder names below are significant. Switch the canonical checkouts to the pinned commits above. From the integration root, expose their source:

```bash
export PYTHONPATH="../survival-env:../survival-agent"
python -B -m examples.custom_policy
python -B -m examples.toy_env_adapter
python -B -m scripts.verify_docs --no-training
```

This workflow was checked against the local sibling checkouts and existing dependencies. It is not an editable-install or clean dependency-resolution test. The README quickstarts in each canonical repository run independently from that repository root. Integration regression subprocesses inherit this source path.

For a separately provisioned virtual environment, `requirements.txt` lists local editable sibling sources; `requirements-language.txt` points to the agent's language dependency definition. Do not confuse dependency declarations with a tested new installation. No install command was run during the split. Wheels/clean builds remain untested; installed setuptools is older than the declared build requirement. No weights, private run artifacts or historical private source were copied into the new repositories.

The agent repository keeps small legacy module forwarding paths for old CLI imports. Those require the corresponding environment or integration source on the path when explicitly invoked. Core model imports and standalone model tests do not require an environment. Environment source never requires agent/ML imports.
