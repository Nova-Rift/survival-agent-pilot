"""Adapt a separate five-arm toy bandit to the existing latent core.

This is a narrow shape/action adapter, not universal environment support.
The randomly initialized model performs no learning or language inference.
"""
import json
import numpy as np
import torch
from survival_agent.integrated import LanguageLatentCore


class ToyBandit:
    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.target = int(self.rng.integers(5))
        self.done = False
        return {'cue': self.target}, {}

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset required')
        if type(action) is not int or not 0 <= action < 5:
            raise ValueError('Expected integer arm 0..4')
        self.done = True
        return {'cue': self.target}, float(action == self.target), True, False, {}


class ToyAdapter:
    @staticmethod
    def encode(observation):
        # The bandit cue is public observation, not privileged state.
        return torch.from_numpy(np.eye(5, dtype=np.float32)[observation['cue']][None])

    @staticmethod
    def decode(action):
        return int(action)


def main():
    torch.set_num_threads(1)
    world = ToyBandit()
    observation, _ = world.reset(seed=3)
    core = LanguageLatentCore(feature_width=5, seed=4).eval()
    with torch.inference_mode():
        features = ToyAdapter.encode(observation)
        logits, value = core.predict(core.represent(features))
        action = ToyAdapter.decode(logits.argmax(-1).item())
    _, reward, terminated, truncated, _ = world.step(action)
    assert features.shape == (1, 5) and logits.shape == (1, 5)
    assert torch.isfinite(value).all() and terminated and not truncated
    print(json.dumps({'action': action, 'reward': reward, 'terminated': terminated,
                      'optimizer_updates': 0, 'trained_result': False}))


if __name__ == '__main__':
    main()
