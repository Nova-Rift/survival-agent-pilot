# Open-loop diagnostic milestone

The original [stage-B smoke](hybrid-stage-b.md) and its seed-0 report are preserved.
No model objective, baseline, live demo or survival mechanic was changed.

[Model-side interpretation and measured limitations](https://github.com/Nova-Rift/survival-agent/blob/hybrid-stage-b/docs/hybrid-stage-b-diagnostics.md)
explains the posterior/prior fork and the evidence. This repository retains the
separate collector/sampler and reproducible experiment:

- [Fixed protocol](experiments/hybrid-stage-b-diagnostic/protocol.json)
- [Full report: all conditions, initial baselines, horizons and calibration bins](experiments/hybrid-stage-b-diagnostic/report.json)
- [All 480 optimizer-update logs and actual target exposures](experiments/hybrid-stage-b-diagnostic/updates.jsonl)
- [Diagnostic command](../examples/hybrid_stage_b_diagnostic.py)

Use model branch `hybrid-stage-b` and this integration branch
`hybrid-stage-b-survival` with sibling `survival-env`. Published installation pins
still select the original extraction commits; they have not been silently changed.
From this integration checkout:

```sh
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m examples.hybrid_stage_b_diagnostic --output runs/hybrid-stage-b-diagnostic
```

Choose a fresh output directory. The fixed plan is written before final data
collection. It trains three model seeds with 80 updates each under prefix and
complete-episode conditions. All optimization finishes before final seeds are
collected/scored. No final scores guide optimizer settings or condition selection.
The broader unchanged-rule world supplies 249 training transitions; disjoint
development and final sets supply 120 each. Training has six deaths and 26
exclusive truncation endpoints; each held-out set has four deaths and 12 exclusive
truncations. Death dominates when both flags occur. The report's counts distinguish
full episode data from the targets actually selected for loss.

The short-prefix condition exposes no death/horizon targets under the broader
configuration. Complete-episode batching remedies that coverage limitation without
joining/padding episodes or dropping final observations. Equal optimizer-update
counts do **not** mean equal supervised-transition or compute budgets.

On final seeds, complete-episode prior reward MSE averages .04718 across three
model seeds, versus .03719 for a training-mean constant. Posterior reward MSE is
.02476. Prior continuation BCE is .16745 versus baseline .16675. At future horizon
five, observed continuation is .75 while priors predict .9380..9476. Good aggregate
accuracy conceals missed deaths. Complete exposure alone is not a demonstrated
fix for open-loop prediction; model initialization variation and posterior/prior
gaps are recorded. Alignment/label fixtures found no code defect requiring an
RSSM-loss change.

The diagnostic used CPU only (about 7.4 seconds), no paid compute or new packages.
Legacy-runtime advisory limitations remain. There is no trained-policy,
survival-improvement or continual-learning result, and no planner was added.
Browser execution and website QA remain separate, unverified work.
