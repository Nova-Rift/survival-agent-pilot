# Model modes, checkpoints and provenance

[Developer index](developers.md) · [Security scope](../SECURITY.md)

| Mode | What exists today |
|---|---|
| Scripted baseline | Random or local greedy policy; no model, training or inference. |
| History agent | Small randomly initialized observation transformer, five-action heads and shallow latent lookahead; not full MuZero. |
| Scratch language backend | Random small language architecture, fixed UTF-8 byte vocabulary; no pretrained assets or claimed language competence. |
| Pretrained language backend | Explicit local Qwen/GPT2-family artifact loading, safetensors only, frozen in supported integrated training. No implicit download. |
| Actor-critic | On-policy survival returns with one-step auxiliary reward/dynamics losses; distinct from MuZero. |
| Minimal MuZero-style | PUCT visit targets, episode replay, n-step returns and recurrent losses. Scratch history encoder trains; optional pretrained feature encoder stays frozen. |

Pretrained fine-tuning is unsupported. No useful learned survival, emergent cooperation, planning advantage or generalization is established. Shared weights do not imply shared private histories.

## Bounded scratch checks

These experiment commands were verified before the split. They are intentionally skipped by the current `--no-training` documentation pass. The regression suite tests update/resume behavior after the split without launching an experiment.

These commands use existing local dependencies and scratch weights, CPU threads bounded, no network or pretrained assets. They perform tiny engineering updates. Use fresh output directories; runtime caps are cooperative and shell timeout is the hard process bound.

```bash
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.muzero_run --output runs/docs-muzero --steps 2 --simulations 2 --updates 1 --max-seconds 8
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.muzero_run --resume runs/docs-muzero/checkpoint.safe --output runs/docs-muzero-resumed --steps 2 --simulations 2 --updates 1 --max-seconds 8
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.training train --config ../survival-agent/survival_agent/configs/from-scratch.json --output runs/docs-ac --horizon 2 --rollout-steps 1 --max-joint-steps 2
OMP_NUM_THREADS=1 timeout 20s python -B -m survival_agent.training train --config ../survival-agent/survival_agent/configs/from-scratch.json --checkpoint runs/docs-ac/checkpoint.safe --output runs/docs-ac-resumed --max-joint-steps 3
```

MuZero reports cumulative optimizer updates and episode counts. Actor-critic's step cap is cumulative on resume: the second command must exceed the saved joint-step count. MuZero replay resume is at episode boundaries. Exact next-update determinism is covered by [test_muzero.py](../tests/test_muzero.py) and [test_training.py](../tests/test_training.py).

## Loading boundaries

[checkpoint_io.py](https://github.com/Nova-Rift/survival-agent/blob/main/survival_agent/checkpoint_io.py) stores schema-validated JSON plus safetensors in a bounded two-member uncompressed archive. It rejects pickle/Torch legacy files, unknown object types, nonfinite scalar metadata, oversized data and unsafe array types. Reconstruction bounds model/world dimensions and validates replay shapes before arithmetic. These guards reduce risk, not a security guarantee.

Only resume checkpoints you control. Cross-version/source changes can intentionally fail provenance checks. This split changes source hashes; earlier source-bound experiment checkpoints may be rejected. No private checkpoints are rewritten. Canonical combined checkpoint code is in `survival_adapters/checkpoints.py`; MuZero CLI checkpoints additionally record a combined collector/world-source hash, while model-only learner state does not depend on environment source. Preserve model specification, backend provenance, source hashes, versions, optimizer/replay/RNG and environment state. Never bypass a mismatch by deleting checks. Do not commit checkpoint contents or private run data.

## Pretrained prerequisites: not run in documentation verification

The fixed official Qwen revision is documented in README and [fetch_qwen.py](../scripts/fetch_qwen.py). The helper requires network access, at least 4 GiB free before starting, and downloads roughly 1 GB of weights plus metadata. Its destination is the ignored project `models/` folder. Verify the generated manifest and license before use. Download duration depends on connection speed; it was not measured in this documentation pass.

The following is a **manual download command**, not a smoke test. It is intentionally skipped by the verifier:

```bash
python -B -m scripts.fetch_qwen
```

The optional `scripts.smoke_language` module performs actual local pretrained generation. It is also excluded from documentation verification; no claim is made that pretrained inference was retested. CLI help can be checked without loading models.
