"""Run a third-party-style scripted policy against the existing world API."""
import json
import numpy as np
from survival_env.simulator import Config, SurvivalWorld


def wait_policy(observations, rng):
    # Policies receive observations, never the world object.
    return np.zeros(len(observations), dtype=np.int64)


def main():
    world = SurvivalWorld(seed=7, config=Config(horizon=3))
    rng = np.random.default_rng(8)
    observation = world.observe()
    total_reward = 0.0
    while not world.done:
        actions = wait_policy(observation, rng)
        observation, reward, done, info = world.step(actions)
        assert observation.shape == (8, 5, 5, 5)
        total_reward += float(reward.sum())
    assert world.tick == 3 and info['truncated'] and done
    assert total_reward == 24.0
    print(json.dumps({'joint_steps': world.tick, 'reward': total_reward,
                      'policy': 'scripted wait', 'trained_result': False}))


if __name__ == '__main__':
    main()
