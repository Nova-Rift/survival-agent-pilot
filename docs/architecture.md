# Architecture and commands

The simulator provides local observations, survival rewards and deterministic collision rules. Model code lives in `survival_agent/`; no source file remains loose in the repository root.

- `agent.py`: small observation-history transformer and shallow latent lookahead.
- `language_backend.py`: local pretrained encoder/generator or random byte-vocabulary alternative. No implicit downloads or APIs.
- `integrated.py`: language features -> latent projection/dynamics/reward/value/policy -> bounded action.
- `training.py`: distinct on-policy actor-critic baseline, data collection and held-out evaluation.
- `muzero.py`: distinct minimal PUCT MCTS, actual visit-count targets, episode replay and recurrent unroll learner. Scratch observation encoding is trainable; the supported pretrained feature path is frozen.
- `checkpoint_io.py`: bounded structured safetensors/JSON checkpoints; no pickle fallback.

Run modules from the repository root, for example `python -m survival_agent.run --help`, `python -m survival_agent.training --help` and `python -m survival_agent.muzero_run --help`. Run tests with `python -m unittest discover -v`.

Language configuration files are in `survival_agent/configs/`. Use `--config survival_agent/configs/from-scratch.json` where supported. Model files live separately in the ignored project-level `models/` directory. The explicit fetch helper runs as `python -m scripts.fetch_qwen`; no runtime starts it automatically.

## Preview provenance

The README GIF is drawn from the actual 129-frame, seed-0 scripted greedy baseline already recorded and shown on the website. Agent positions, food, hunger and survival counts come from that trace. It does not depict trained model behavior or cooperation evidence. `assets/preview-provenance.json` records the input hash and image dimensions.

To render another trusted numerical trace, install Pillow in a separate environment and run `python -m scripts.render_preview --frames path/to/frames.json --output assets/survival-preview.gif`. Rendering makes no model or network calls.

This is not full MuZero, useful survival learning, validated language cooperation or continual-learning evidence. See README and SECURITY for limits.
