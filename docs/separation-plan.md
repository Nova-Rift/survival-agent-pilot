# Planned environment / agent separation

[Developer index](developers.md) · [Current interfaces](interfaces.md)

**Proposal only.** The packages and contracts below are not implemented or installable today. Keep the existing repository and forwarding modules to preserve links, history and commands.

1. `survival_env`: NumPy-only mechanics, optional local communication and recording.
2. `survival_agent`: models, bounded encoders, search, learning and model state; no concrete simulator imports.
3. Adapters and experiments: environment construction, observation/action conversion, collection and combined resume. The two cores do not depend on each other.

First remove world reconstruction from agent checkpoints and world creation from MuZero/actor-critic collectors. Move survival-specific language formatting to adapters. Action heads, searches and input encoding currently contain survival dimensions; these need explicit bounded specifications before broader compatibility can be claimed.

Target a simultaneous multiagent interface: reset returns observation/info dictionaries; step accepts current-agent actions and returns observation/reward/termination/truncation/info dictionaries. Stable possible-agent IDs, changing active agents, local observations, one simultaneous joint transition, death termination, survivor horizon truncation and explicit seeded resets must be documented and tested. Final observations must survive the boundary for bootstrap targets.

This matches the shape of [PettingZoo's Parallel API](https://pettingzoo.farama.org/api/parallel/), but compatibility is **not established**. A future optional wrapper must expose spaces and pass its [parallel API and seed tests](https://pettingzoo.farama.org/content/environment_tests/). [Gymnasium](https://gymnasium.farama.org/api/env/) is single-agent; a separate wrapper would need an explicit controlled-agent/opponent policy choice. No new dependency has been added.

Acceptance: an external scripted policy runs the environment without ML imports; an agent works through another environment's explicit encoder/action adapter; core-agent imports cannot reach the simulator; seeds, inactive agents, termination/truncation and checkpoint security regressions pass. Initial support should state finite discrete-action limits. This proposal does not promise arbitrary environments, continuous controls or trained-policy transfer.

Potential richer-engine adapters remain unselected and unimplemented. Any future integration must define local perception filtering, nearby communication, separate per-agent state, action mapping, and whether the world pauses or advances while decisions are pending. No Minecraft, Mineflayer, Luanti or Craftium support is claimed.
