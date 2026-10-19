"""Single-agent vector adapter; canonical mechanics remain in survival-env.

No autoreset, privileged policy input, model copy, new mechanics or Gym dependency.
"""
from dataclasses import asdict
import numpy as np
from survival_env.simulator import Config, SurvivalWorld, greedy_policy
from survival_agent.hybrid.contracts import ActionSpec, EnvSpec, VectorAdapter
from survival_agent.hybrid.replay import SequenceReplay


class SurvivalVectorEnv:
    def __init__(self, config=Config(size=6, agents=1, radius=1, max_hunger=4,
                                   initial_food=3, food_per_step=0, horizon=8)):
        if not isinstance(config, Config) or config.agents != 1:
            raise ValueError('Hybrid stage B supports exactly one survival agent')
        self.spec = EnvSpec(5*(2*config.radius+1)**2, ActionSpec('discrete', count=5))
        self.config = config
        self.world = None

    def reset(self, seed=None):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Explicit bounded seed required')
        self.world = SurvivalWorld(seed, self.config)
        return self.world.observe()[0].reshape(-1).copy(), {}

    def step(self, action):
        if self.world is None:
            raise RuntimeError('Reset before stepping')
        action = self.spec.action.validate(action)
        obs, rewards, _, info = self.world.step([action])
        # Keep the actual final observation; construction of the next world is explicit.
        return obs[0].reshape(-1).copy(), float(rewards[0]), bool(info['died'][0]), bool(info['truncated']), {}

    def close(self):
        pass


def collect_replay(*, episodes=16, seed=0, config=None):
    if type(episodes) is not int or not 1 <= episodes <= 32 or type(seed) is not int or not 0 <= seed < 2**32-32:
        raise ValueError('Bounded collection budget required')
    env = SurvivalVectorEnv() if config is None else SurvivalVectorEnv(config)
    if env.config.horizon > 128:
        raise ValueError('Stage B collection horizon capped at 128')
    adapter = VectorAdapter(env, env.spec)
    replay = SequenceReplay(env.spec, seed=seed)
    rng = np.random.default_rng(seed)
    summaries = []
    try:
        for episode in range(episodes):
            observation, _ = adapter.reset(seed=seed+episode)
            replay.start(observation)
            total_reward, steps = 0., 0
            while True:
                if episode % 2:
                    width = 2*env.config.radius+1
                    action = int(greedy_policy(observation.reshape(1, 5, width, width), rng)[0])
                else:
                    action = int(rng.integers(5))
                transition = adapter.step(action)
                replay.append(transition.action, transition.reward, transition.next_observation,
                              transition.terminated, transition.truncated)
                total_reward += transition.reward
                steps += 1
                observation = transition.next_observation
                if transition.terminated or transition.truncated:
                    break
            summaries.append({'seed': seed+episode, 'steps': steps, 'reward': total_reward,
                              'collector': 'local_greedy' if episode % 2 else 'random',
                              'terminated': transition.terminated, 'truncated': transition.truncated})
    finally:
        env.close()
    return replay, {'config': asdict(env.config), 'episodes': summaries,
                    'real_steps': sum(s['steps'] for s in summaries)}
