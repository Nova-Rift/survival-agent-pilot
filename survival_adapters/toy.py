"""Explicit five-arm cue/action adapter; no survival simulator import."""
import numpy as np
import torch

class ToyAdapter:
    @staticmethod
    def encode(observation):
        # The bandit cue is public observation, not privileged state.
        return torch.from_numpy(np.eye(5, dtype=np.float32)[observation['cue']][None])

    @staticmethod
    def decode(action):
        return int(action)
