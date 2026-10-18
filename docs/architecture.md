# Architecture and directory map

[Developer index](developers.md) · [Current interfaces](interfaces.md) · [Planned split](separation-plan.md)

The simulator, agents and experiment runners currently share `survival_agent/`. This is the actual layout; independent environment/agent packages are planned.

| Path | Responsibility |
|---|---|
| [simulator.py](../survival_agent/simulator.py) | Local observations, seeded world transitions, collision rules, random/greedy baselines. |
| [communication.py](../survival_agent/communication.py) | Optional local plain-text transport. |
| [agent.py](../survival_agent/agent.py) | Small history transformer and shallow latent lookahead. |
| [language_backend.py](../survival_agent/language_backend.py) | Local pretrained or random byte-vocabulary language mode; no implicit downloads. |
| [integrated.py](../survival_agent/integrated.py) | Frozen features to trainable latent model and bounded actions. |
| [training.py](../survival_agent/training.py) | On-policy actor-critic collection, updates and held-out evaluation. |
| [muzero.py](../survival_agent/muzero.py) | Minimal PUCT MCTS, visit targets, episode replay and recurrent learner. |
| [checkpoint_io.py](../survival_agent/checkpoint_io.py) | Structured safetensors/JSON serialization and bounds. |
| [configs/](../survival_agent/configs/) | Explicit scratch/pretrained configuration. |
| [examples/](../examples/) | Small runnable custom-policy and toy-adapter examples. |
| [tests/](../tests/) | Correctness, exact resume, privacy and security regressions. |
| [scripts/](../scripts/) | Explicit download, rendering and docs verification utilities. |
| [assets/](../assets/) | Recorded README GIF and provenance. |

Model files belong in the ignored project-level `models/` folder. Checkpoints and run datasets belong in ignored `runs/` folders, not Git. See [model modes](models.md) before loading anything.

## Inspect the CLI without loading a model

```bash
python -B -m survival_agent.run --help
python -B -m survival_agent.training --help
python -B -m survival_agent.muzero_run --help
python -B -m survival_agent.language_run --help
python -B -m survival_agent.integrated_run --help
```

## Render a recorded trace

Requires Pillow, an optional rendering tool already available in the verification runtime. Pillow installation in a new environment was not tested. First run the README baseline command; then:

```bash
python -B -m scripts.render_preview --frames runs/baseline/frames.json --output runs/preview.gif
```

The README's existing GIF comes from the actual seed-0, 129-frame scripted greedy trace. [Its provenance](../assets/preview-provenance.json) records the source hash and dimensions. Rendering another trace makes no model/network calls and writes its own sibling provenance file. It does not establish trained behavior or cooperation.
