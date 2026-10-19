# Synthetic survival v2 — planned, not implemented

The completed legacy data diagnostic is frozen. Its environment code, data hashes
and historical reports remain unchanged. New mechanics must use a separate
environment/version identifier; do not rename old results as synthetic-v2.

Proposed pressures are finite operating power, cumulative integrity damage,
unstable environmental zones, cascading system faults and finite operational
lifetime. Legacy depletion/replenishment is an existing resource mechanic;
additional subsystems are not implemented by relabeling it. Death means
irreversible loss of operation for that episode. No forced-cooperation bonus.

Required lifetime contract for the next version:

- A configured finite `wear_limit >= 2`, measured in simulation transitions.
  Every valid step for a living agent increments irreversible wear monotonically,
  including waiting and blocked moves. Wall-clock time, browser throttling and
  display frame rate never affect wear or reward.
- Recharging replenishes operating power; repairs may restore integrity, but
  neither can decrease wear or raise/reset the episode's lifetime limit.
- At the limit, terminal wear failure is guaranteed even with unlimited recharge
  and repair. It sets termination true and continuation zero. Dead state absorbs;
  an explicit new episode initializes a new agent lifetime, not a resurrection
  within the old episode. Checkpoint/resume preserves wear and the configured limit.
- An earlier experiment horizon truncates a still-operational agent and retains
  bootstrapping from its true final observation. If wear failure and horizon occur
  together, retain true termination and continuation zero. Never reward time spent
  waiting outside simulation stepping.
- Keep the current post-transition alive-step reward convention initially: reward
  one only when operational after the step, zero on terminal failure. Thus the
  no-other-failure return ceiling is `min(horizon, wear_limit-1)`. Report raw return,
  normalized ceiling fraction, attempted steps, failure cause and truncations.
  Make remaining lifetime observable for initial controlled tests rather than
  silently introducing another hidden-state confound.

Before implementation, specify deterministic update order and observations for
damage/zones/faults, then add monotonic-wear, recharge/repair non-reset, terminal
limit, horizon precedence, saved-final-observation and exact-resume tests.
No new environment mechanics or trained-policy footage exist in this design step.
