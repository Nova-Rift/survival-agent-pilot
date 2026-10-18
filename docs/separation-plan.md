# Separation status and next steps

[Developer index](developers.md) · [Current interfaces](interfaces.md)

## Implemented source separation

- `survival_env`: NumPy-only simulator, optional local communication, recorded replay and baseline runner. No agent/model imports.
- `survival_agent`: history/latent models, local language backend, search, replay learner and structured model serialization. Core imports do not load the concrete simulator.
- `survival_adapters`: survival-specific collection, experiment runners, combined world/model checkpoints and an explicit toy-environment adapter.
- Existing `survival_agent` simulator/runner imports and CLI paths forward to the new implementation. Legacy checkpoint helper functions lazily call the combined checkpoint adapter.
- The environment has its own NumPy-only dependency definition; the agent has its own model/language dependencies. This pilot repository contains only adapters/examples/tests and pins canonical source commits. See [local workflow](repositories.md). Fresh installation/resolution was not tested.

[Boundary tests](../tests/test_portability.py) block ML/agent imports when running the environment and block the survival simulator when running models plus the separate toy adapter. They prove usable import boundaries, not universal task compatibility. Existing world behavior and fixed five-action model semantics remain unchanged.

## Still planned

An environment-independent collector protocol, configurable action heads/observation encoders, survival-specific language formatting extraction, remain future work. Current survival experiment adapters still assume eight agents, fixed local grid semantics and five actions.

A future simultaneous multiagent wrapper would expose seeded reset plus dictionary-based observations, actions, rewards, termination, truncation and info; stable possible-agent IDs; changing active agents; and final observations for bootstrap targets. The existing world still uses construction to reset and a four-value step result. No new reset API was invented for this increment.

[PettingZoo's Parallel API](https://pettingzoo.farama.org/api/parallel/) is a possible target, but compliance is not established. A wrapper must expose spaces and pass its [parallel API and seed tests](https://pettingzoo.farama.org/content/environment_tests/). [Gymnasium](https://gymnasium.farama.org/api/env/) is single-agent and would require an explicit opponent/control policy. No standards dependency was added.

Richer engines remain unselected and unimplemented. Any future integration must define local perception filtering, nearby communication, separate per-agent state, action mapping, and whether the world pauses or advances while decisions are pending. No Minecraft, Mineflayer, Luanti or Craftium support, arbitrary-environment support, continuous controls or trained-policy transfer is claimed.
