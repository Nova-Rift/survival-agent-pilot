"""Bounded diverse real survival replay and exact-context fixed-loss-budget batches."""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import numpy as np
from survival_env.simulator import Config, greedy_policy
from survival_agent.hybrid.contracts import VectorAdapter
from survival_agent.hybrid.replay import SequenceReplay
from .hybrid_survival import SurvivalVectorEnv


CONFIGURATIONS = (
    Config(size=8, agents=1, radius=1, max_hunger=8, initial_food=2, food_per_step=0, horizon=24),
    Config(size=8, agents=1, radius=1, max_hunger=12, initial_food=10, food_per_step=1, horizon=24),
    Config(size=6, agents=1, radius=1, max_hunger=12, initial_food=12, food_per_step=2, horizon=24),
    Config(size=10, agents=1, radius=1, max_hunger=16, initial_food=12, food_per_step=1, horizon=32),
)
POLICIES = ('random', 'greedy', 'wait', 'epsilon_greedy')


def collect_diverse(*, episodes, seed_base):
    if type(episodes) is not int or not 1 <= episodes <= 384 or type(seed_base) is not int or not 0 <= seed_base < 2**32-384:
        raise ValueError('Bounded collection request required')
    rng = np.random.default_rng(seed_base)
    spec = SurvivalVectorEnv(CONFIGURATIONS[0]).spec
    replay = SequenceReplay(spec, seed=seed_base, capacity=16384, max_bytes=16*1024*1024)
    summaries = []
    for index in range(episodes):
        # Choose map/resource config and policy BEFORE the outcome. Never resample
        # to balance deaths or filter inconvenient final/evaluation episodes.
        configuration = int(rng.integers(len(CONFIGURATIONS)))
        policy = POLICIES[int(rng.integers(len(POLICIES)))]
        env = SurvivalVectorEnv(CONFIGURATIONS[configuration])
        adapter = VectorAdapter(env, spec)
        observation, _ = adapter.reset(seed=seed_base+index)
        replay.start(observation)
        steps, total_reward = 0, 0.
        try:
            while True:
                if policy == 'wait':
                    action = 0
                elif policy == 'random' or (policy == 'epsilon_greedy' and rng.random() < .25):
                    action = int(rng.integers(5))
                else:
                    action = int(greedy_policy(observation.reshape(1, 5, 3, 3), rng)[0])
                transition = adapter.step(action)
                replay.append(transition.action, transition.reward, transition.next_observation,
                              transition.terminated, transition.truncated)
                observation = transition.next_observation
                steps += 1
                total_reward += transition.reward
                if transition.terminated or transition.truncated:
                    break
            summaries.append({'seed': seed_base+index, 'configuration': configuration, 'policy': policy,
                              'steps': steps, 'reward': total_reward,
                              'terminated': transition.terminated, 'truncated': transition.truncated})
        finally:
            env.close()
    if len(replay.episodes) != episodes:
        raise AssertionError('Data collection exceeded capacity: no silent episode eviction allowed')
    return replay, {'configurations': [asdict(c) for c in CONFIGURATIONS],
                    'policies': list(POLICIES), 'episodes': summaries,
                    'real_steps': sum(e['steps'] for e in summaries),
                    'deaths': sum(e['terminated'] for e in summaries),
                    'exclusive_truncations': sum(e['truncated'] and not e['terminated'] for e in summaries),
                    'both': sum(e['truncated'] and e['terminated'] for e in summaries)}


def replay_digest(replay):
    digest = hashlib.sha256()
    for episode in replay.episodes:
        for key in sorted(episode):
            digest.update(key.encode())
            value = episode[key]
            if value is not None:
                digest.update(value.dtype.str.encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def fixed_target_batch(replay, rng, *, batch_size=4, unroll=4):
    """Equal supervised-transition count, variable exact prefix/burn-in cost.

    Pick length groups proportional to episode count, then a shared end index:
    half endpoint-focused, half uniform. All conditions use the identical rule.
    Reconstruct from the actual episode start; never approximate interior context.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 32 or type(unroll) is not int or not 1 <= unroll <= 32:
        raise ValueError('Invalid fixed-target batch bounds')
    eligible = [e for e in replay.episodes if unroll <= len(e['actions']) <= 128]
    if not eligible:
        raise ValueError('No episodes fit the fixed target unroll')
    length = len(eligible[int(rng.integers(len(eligible)))]['actions'])
    group = [e for e in eligible if len(e['actions']) == length]
    end = length if rng.random() < .5 else int(rng.integers(unroll, length+1))
    rows = []
    for _ in range(batch_size):
        episode = group[int(rng.integers(len(group)))]
        row = {key: None if value is None else value[:end+(key in ['observations', 'is_first', 'masks'])].copy()
               for key, value in episode.items()}
        row.update(loss_mask=np.arange(end) >= end-unroll,
                   context_is_episode_start=True, episode_offset=0)
        rows.append(row)
    return rows


def first_episodes(replay, count):
    if type(count) is not int or not 1 <= count <= len(replay.episodes):
        raise ValueError('Invalid training-only subset size')
    subset = SequenceReplay(replay.spec, capacity=16384, max_bytes=16*1024*1024)
    subset.episodes = deepcopy(replay.episodes[:count])
    return subset
