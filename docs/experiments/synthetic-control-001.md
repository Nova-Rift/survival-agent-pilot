# First synthetic-v2 posterior-control run — October 19, 2026

A compatible trained policy now executes in the exact JavaScript environment
published on clarkwhitehead.com. It underperforms every held-out reference and is
not the website controller. The live homepage correctly remains labeled scripted.
This is a bounded engineering run, not a useful-control or continual-learning result.

The Python adapter uses a local newline-JSON Node bridge to the canonical
`survival-env/browser/synthetic.mjs` at commit
`e4d3f66a3f870fefe7340e35f206b587eb0c69e4`. No rules are duplicated in Python.
All eight agents have identical shared weights, independent posterior memory and
only local 8×5×5 observations. The observer's overhead state is not a model input.

The existing small categorical RSSM/GRU is fitted on 9,453 real active-agent
transitions from 16 predetermined worlds, half random and half scripted, for 300
updates. Its weights are then frozen. The existing linear posterior actor/critic
and boundary-aware survival targets are adapted to mask inactive agent slots.
The actor learns on real interactions only: 128 updates, 2,048 joint steps,
10,729 active-agent transitions. No failed prior planning, CBP, GPU, paid inference,
new package or environment redesign is involved. CPU run time was about 18 seconds;
paid cost was $0.

## Frozen selection and held-out results

Training worlds start at seed110000; world-model data use30000–30015. Three
checkpoints (updates32/64/128) were ranked on eight fixed validation worlds
220000–220007. All had mean26.265625; the predetermined earliest-tie rule selected
update32. The safe checkpoint hash and selection were saved BEFORE reporting
sixteen final worlds330000–330015. Final worlds were not used to choose or retune.
These are world seeds, not 128 independent agents: agents compete within worlds.
Only one training seed was run.

| Controller | Mean alive-step return per agent |
|---|---:|
| Trained posterior actor | 24.453125 |
| Identical untrained attachment | 25.8046875 |
| Random | 63.0234375 |
| Local scripted greedy / hazard avoidance | 78.6015625 |

Wear forces termination on transition96. Since the environment pays zero on the
death transition, the return ceiling is95, not96. The fixed representative
recording seed70000 returned25.125 per agent. Safe-checkpoint reload reproduced
all frames and scores exactly. No representative seed was selected after outcomes.
The small frozen posterior/linear attachment is insufficient in this run; do not
claim learning improvement, useful planning, cooperation or retention.

## Reproduction and preserved artifacts

From this checkout with sibling environment/model checkouts and the existing
PyTorch/NumPy runtime:

```sh
PYTHONPATH=.:../survival-env:../survival-agent python3 examples/train_synthetic_control.py --environment ../survival-env/browser --output /tmp/fresh-synthetic-control-run
```

Use a fresh output directory. The script verifies the exact clean environment
commit, records implementation hashes, writes its protocol before collecting data,
selects before final evaluation, and saves the checkpoint and fixed recording.
Checkpoints/recordings remain private and are not included here. A saved selected
policy contains model weights, not full optimizer/environment training-resume state.

- [Protocol](synthetic-control-001/protocol.json)
- [Frozen selection](synthetic-control-001/selection-frozen.json)
- [Full per-world results](synthetic-control-001/report.json)

All34 model tests and64 integration tests passed, including head gradients and
frozen RSSM, exact Node environment/reset behavior, inactive/terminal return masks,
real weight updates and separate evaluation-world state. All11 synthetic and24
legacy JavaScript tests also passed. Website build, HTML controls, assets and five
pinned module hashes passed; actual browser layout/hydration remains unverified.
