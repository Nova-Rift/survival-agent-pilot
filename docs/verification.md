# Documentation verification

[Developer index](developers.md) · [Security scope](../SECURITY.md)

Run from the repository root:

```bash
python -B -m scripts.verify_docs
```

The [verifier source](../scripts/verify_docs.py) inventories every command in `bash` blocks across README and `docs/*.md`, validates local Markdown targets, and invokes an explicit allowlist without shell evaluation. Commands use the current Python interpreter, offline Hugging Face settings and one CPU thread. Each run redirects `runs/` paths into a fresh temporary directory and removes outputs afterward. Identical commands are reported as duplicates. Unknown commands fail until reviewed.

## Scope

| Item | Verification |
|---|---|
| README baseline | Run the documented 128-step greedy command; check summary/frame consistency and generated replay HTML. |
| Developer quickstart | Run four-step baseline and verify all three output files. |
| Custom policy | Execute complete example, assert three steps/reward 24; repeat with Torch/Transformers/safetensors imports blocked. |
| Toy adapter | Execute randomly initialized five-action core against separate toy bandit; verify valid action, finite value and terminal result. |
| CLI navigation | Run help for baseline, actor-critic, MuZero, language and integrated runners without loading models. |
| Actor-critic | Run README collect/train/evaluate, verify checkpoint/data/report output and all four held-out comparisons finish. |
| MuZero | Run README bounded update and separate two-step update/resume; verify cumulative updates increase to two. |
| Actor-critic resume | Run two steps, resume to cumulative step three and check the report. |
| Rendering | Generate GIF from the temporary README trace; check dimensions, every frame and provenance. Requires existing Pillow. |
| Regression suite | Execute the full unittest command. Exact next-update checkpoint determinism, rejection boundaries and replay escaping are covered there. |
| Local links | Check all relative Markdown targets in README and docs exist, including source links and directory navigation. |

The examples exercise real local scratch-model forward/backward passes where applicable. They are engineering checks, not long training or evidence of a useful learned policy. No pretrained model inference, GPU, API or paid resource is used.

## Explicitly excluded

- Creating/activating a new virtual environment and installing requirements: not exercised. A clean resolved environment needs network access and disk space for ML wheels; exact resolver size/time has not been measured. No clean-start compatibility claim is made.
- The manual pinned Qwen download: not run. It requires at least 4 GiB free and roughly 1 GB of model downloads; time depends on connection speed.
- Pretrained generation and the optional pretrained smoke module: not retested.
- Installing Pillow, external engine adapters, PettingZoo/Gymnasium compliance, browser rendering and every external website: not tested by this script. The current code has no standards wrapper.
- A fully resolved transitive dependency/security audit: not performed. Functional passing tests are not a security certification.

The verification runtime uses Python 3.10.12, NumPy 1.26.4, PyTorch 2.2.0+cu121, Transformers 4.37.2 and safetensors 0.4.5, on CPU. This legacy runtime has known advisories; no global packages were upgraded. The recommended newer requirements are explicitly untested together.

## Latest documentation pass

See the verifier's JSON output for each exact command, source file/line and passed/skipped/duplicate status. The publication pass is recorded after verification below; rerun after changing any command or example. Keep all runnable shell examples in `bash` fences so they stay in the inventory.

Publication check: 20 distinct bounded commands passed, two repeated example commands reused their passing result, and nine command occurrences were explicitly skipped (six setup instructions, one model-download instruction and two verifier self-invocations). The regression suite passed all 48 tests. Both checkpoint-resume examples, replay generation, the custom policy and toy adapter passed. [Machine-readable results](verification-results.json) contain the exact inventory; temporary run artifacts were removed. These results apply to the legacy runtime above, not a fresh installation.
