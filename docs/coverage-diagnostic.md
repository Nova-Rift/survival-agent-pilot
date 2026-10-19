# Larger data-coverage diagnostic

[Interpretation, matched budgets and decision](https://github.com/Nova-Rift/survival-agent/blob/hybrid-stage-b/docs/coverage-diagnostic.md)
· [Full machine-readable report](experiments/hybrid-coverage-diagnostic/report.json)
· [Planned protocol](experiments/hybrid-coverage-diagnostic/planned_protocol.json)
· [Frozen protocol](experiments/hybrid-coverage-diagnostic/frozen_protocol.json)
· [2,700 update logs](experiments/hybrid-coverage-diagnostic/updates.jsonl).

`survival_adapters/coverage_data.py` owns bounded diverse real collection and
exact-prefix fixed-target batching. The core model remains independently owned in
`survival-agent`. Legacy mechanics are unchanged. No evaluation episodes are
resampled/filtered to balance deaths; config/policy choices precede outcomes.
Training collects 7,378 transitions with 189 deaths and 195 surviving truncations.
Final reporting covers 128 fresh episodes with 68 deaths and 60 surviving
truncations, after protocol/model hashes are frozen. Previous final reports remain
unchanged. No data from final seeds select parameters or an architecture.

Run with the two sibling model/environment checkouts and a fresh output path:

```sh
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 180s python -B -m examples.hybrid_coverage_diagnostic --output runs/hybrid-coverage-diagnostic
```

Fixed architecture and 300 updates / 4,800 supervised targets per condition/seed,
three seeds, three dataset conditions. Context reconstruction work differs and is
reported. The larger dataset does not establish adequately calibrated prior death
prediction: all full-data models miss deaths at threshold .5 and remain close to
the trivial reward-mean baseline. No planner is justified by this result.

The run took about 63.6 seconds on CPU. No packages, paid compute or GPU. Source
hashes, replay digests, runtime, calibration and per-horizon counts are recorded.
The report is minified JSON to reduce retained size; no fields are omitted.
Checkpoints, private paths and weights are not published. No survival/retention
claim. Synthetic-v2 design and demo eligibility are separate notes, not implemented
mechanics or trained policy footage.
