# Evaluated-policy recording publication plan

The homepage remains the fixed-seed scripted greedy recording. Stage B trains a
world model, not a policy; prediction loss cannot qualify it for a featured-model
update. No trained footage or new scores are fabricated.

For each future candidate: bind the policy checkpoint hash, model source commit,
environment ID/source commit/config and training budget. Evaluate actual frozen
policy behavior on a predeclared validation suite against random and local-greedy
baselines, with survival/return per seed and aggregate uncertainty. Select by
validation survival-policy performance, never world-model fitting loss. Freeze
the selected checkpoint/protocol before fresh final tests; final results are
reporting evidence, not a model-selection signal.

Record the selected policy on a representative seed fixed **before** viewing its
episode (initial recording seed 70000). Use the bird's-eye observer recording;
the policy still receives only local observations. Retain a complete run including
failures, not the best-looking episode. No live inference is needed for playback.
For synthetic-v2 report the configured wear ceiling and normalize scores correctly;
do not compare unnormalized results across different environment versions.

Before replacing homepage media: validate replay provenance, exact checkpoint,
labels, seed count, baseline comparison and complete-run hashes; retain the prior
recording/version. Keep the homepage compact with a short status label and a link
to detailed evidence. Clark authorizes future publication when a real evaluated
policy meets these conditions. This document is a workflow plan, not an implemented
automatic publisher or scheduled monitoring job. No eligible policy exists yet.
