# Current interfaces and adapters

[Developer index](developers.md) · [Source map](architecture.md) · [Separation status](separation-plan.md)

## Existing environment API

See [simulator.py](https://github.com/Nova-Rift/survival-env/blob/main/survival_env/simulator.py). `SurvivalWorld(seed=0, config=Config())` initializes an episode; there is currently **no `reset()` method**. Construct another world to restart. `observe()` returns a fresh `float32` array of shape `(agents, 5, 2*radius+1, 2*radius+1)`, normally `(8, 5, 5, 5)`.

Channels are food, self-at-center, another living agent, valid in-bounds cell, and own hunger divided by maximum hunger. Dead agents have all-zero observations. Policies must not read world positions, food maps or other agents' histories. `frame()` and `state()` are privileged replay/checkpoint tools.

`step(actions)` accepts one integer action per configured agent, including inactive slots: 0 wait, 1 north, 2 east, 3 south, 4 west. Dead agents' valid actions are ignored. The return value is `(next_observations, rewards, done, info)`, **four values**, not Gymnasium's five. Rewards are one for each agent alive after the transition, otherwise zero. `info['died']` and `info['ate']` are boolean per-agent arrays; `info['truncated']` is a scalar horizon flag. `done` means horizon reached or all agents dead. Calling step after done raises `RuntimeError`.

Moves resolve simultaneously. Occupied destinations, swaps and contested empty destinations fail. Hunger decreases; food can save an agent before death is decided; food then spawns. Death is absorbing. For learning, derive per-agent termination from `info['died']`; truncation is the horizon flag AND that agent still alive. Death on the horizon step is terminal. Preserve the final observation and bootstrap only nonterminal transitions.

For deterministic comparison, use equal world seeds, equal actions and equal configuration. Policy RNG is separate. Checkpoint resume must preserve both streams; constructing a new world is not a substitute for restoring its state.

## Write a custom scripted policy

Use the complete runnable [custom_policy.py](../examples/custom_policy.py). Its callable is `policy(observations, rng) -> int64[agents]`. Replace the wait policy with your own observation-only rule and keep valid actions for every slot. It imports no ML library.

```bash
python -B -m examples.custom_policy
```

This callable is the existing baseline runner convention, not a framework-wide plugin registry. For now, write your own rollout loop as the example does; the baseline CLI accepts only its built-in random/greedy choices.

## Use an agent component with another environment

The runnable [toy_env_adapter.py](../examples/toy_env_adapter.py), using [the explicit toy adapter](../survival_adapters/toy.py), defines a separate one-step, five-arm bandit. Its adapter converts the public cue to a five-dimensional feature vector and decodes a scalar action. `LanguageLatentCore(feature_width=5)` accepts `(batch, 5)` features, produces a latent representation, then `(batch, 5)` policy logits and `(batch,)` values. Despite its name, this class does not load a language model.

```bash
python -B -m examples.toy_env_adapter
```

The example does not prove a whole trainer can switch environments. Survival-specific collectors now live in `survival_adapters` and still create `SurvivalWorld`, use eight agents, and depend on survival formatting. `HistoryAgent` expects history tensors `(batch, history, 5, obs_width, obs_width)` and five actions; the standard path uses history 4 and width 5. Language controllers additionally require the default local-grid semantics. Other observations need an honest task-specific encoder; other action counts or continuous controls need model changes.

Optional [LocalCommunication](https://github.com/Nova-Rift/survival-env/blob/main/survival_env/communication.py) is environment-side transport: bounded plain text, local range, next-step expiry and delivery only to survivors. It does not execute messages, shape rewards or expose world coordinates to policies.

## Implemented dependency boundaries

Importing `survival_env` and running its baseline does not import `survival_agent`, Torch, safetensors or Transformers. Model/search imports and the toy adapter do not import the concrete survival simulator. [Fresh-process tests](../tests/test_portability.py) enforce both by raising on forbidden imports. Combined checkpoints and collection are explicitly adapter operations. Old `survival_agent.simulator` and runner paths are compatibility shims, so callers using those integration paths intentionally reach the environment.
