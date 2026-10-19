"""Actual on-policy control learning; reuses the existing boundary-aware targets."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from copy import deepcopy
import time
import numpy as np
import torch
from torch.nn import functional as F
from survival_env.simulator import Config, SurvivalWorld, greedy_policy
from survival_agent.hybrid.world_model import WorldModel, ModelConfig, WorldModelTrainer
from survival_agent.hybrid.control import PosteriorActorCritic
from survival_agent.hybrid import world_model as world_module, control as control_module
from survival_agent import checkpoint_io
from .training import survival_targets
from .coverage_data import collect_diverse, fixed_target_batch, replay_digest


CONFIG = Config(size=6, agents=1, radius=1, max_hunger=8, initial_food=5, food_per_step=1, horizon=32)
PROTOCOL = {'model': 'frozen categorical RSSM posterior + linear actor/value heads',
    'environment': asdict(CONFIG), 'environment_version': 'legacy-gridworld-resource-depletion',
    'world_model_seed': 0, 'world_model_real_transitions': 7378, 'world_model_updates': 300,
    'policy_seed': 123, 'policy_real_steps': 16384, 'parallel_worlds': 4, 'rollout_steps': 16,
    'checkpoints': [4096, 8192, 16384], 'validation_seeds': list(range(200000, 200016)),
    'final_seeds': list(range(300000, 300016)), 'recording_seed': 70000,
    'selection': 'highest validation mean alive-step return among TRAINED checkpoints; earliest step breaks ties',
    'gamma': .99, 'target_ema': .99, 'learning_rate': .001, 'entropy_weight': .01,
    'policy_representation': 'categorical mode posterior updated on every real observation',
    'planning': False, 'recycling': False, 'synthetic_wear_or_hazards': False, 'cost_usd': 0}


def weights_digest(agent):
    digest = hashlib.sha256()
    for key, value in agent.state_dict().items():
        digest.update(key.encode()); digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def first_state(agent, observation):
    model = agent.world_model
    with torch.no_grad():
        return model.observe_step(model.initial(1), torch.zeros(1, dtype=torch.int64),
            torch.from_numpy(observation.reshape(1, -1)), torch.ones(1, dtype=torch.bool), stochastic=False)[0]


def evaluate(agent, seeds, *, method='trained', record=False):
    if method not in ['trained', 'untrained', 'random', 'greedy']:
        raise ValueError('Unknown real evaluation policy')
    rows, recordings = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for seed in seeds:
            world = SurvivalWorld(seed, CONFIG); rng = np.random.default_rng(seed+11)
            state = first_state(agent, world.observe())
            frames = [world.frame()] if record else []
            total = 0.
            while not world.done:
                if method == 'random':action = int(rng.integers(5))
                elif method == 'greedy':action = int(greedy_policy(world.observe(), rng)[0])
                else:
                    features = agent.world_model.features(state)
                    logits, _ = agent.predict(features)
                    action = int(logits.argmax(-1)[0])
                obs, reward, _, info = world.step([action])
                total += float(reward[0])
                state, _, _ = agent.world_model.observe_step(state, torch.tensor([action]),
                    torch.from_numpy(obs.reshape(1, -1)), torch.zeros(1, dtype=torch.bool), stochastic=False)
                if record:frames.append(world.frame())
            rows.append({'seed': seed, 'alive_step_return': total, 'steps': world.tick,
                         'terminated': bool(info['died'][0]), 'truncated': bool(info['truncated']),
                         'resources_consumed': None})
            if record:recordings.append(frames)
    return {'method': method, 'seed_count': len(seeds), 'episodes': rows,
            'mean_alive_step_return': float(np.mean([r['alive_step_return'] for r in rows])),
            'seconds': time.perf_counter()-start}, recordings


class OnlinePosteriorTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=.001)
        self.action_rng = torch.Generator().manual_seed(123)
        self.next_seed = 100000
        self.worlds = [self._world() for _ in range(4)]
        with torch.no_grad():
            self.state, _, _ = agent.world_model.observe_step(agent.world_model.initial(4), torch.zeros(4, dtype=torch.int64),
                torch.from_numpy(np.stack([w.observe().reshape(-1) for w in self.worlds])), torch.ones(4, dtype=torch.bool), stochastic=False)
        self.steps, self.updates = 0, 0

    def _world(self):
        if self.next_seed >= 200000:raise AssertionError('Training seed range exhausted')
        result = SurvivalWorld(self.next_seed, CONFIG); self.next_seed += 1
        return result

    @torch.no_grad()
    def collect(self):
        data = {key: [] for key in ['features', 'actions', 'rewards', 'terminated', 'truncated', 'next_values']}
        model = self.agent.world_model
        for _ in range(16):
            features = model.features(self.state).detach().clone()
            logits, _ = self.agent.predict(features)
            actions = torch.multinomial(logits.softmax(-1), 1, generator=self.action_rng).squeeze(-1)
            next_obs, rewards, terminated, truncated = [], [], [], []
            for i, world in enumerate(self.worlds):
                obs, reward, _, info = world.step([int(actions[i])])
                next_obs.append(obs.reshape(-1)); rewards.append(float(reward[0]))
                terminated.append(bool(info['died'][0])); truncated.append(bool(info['truncated']))
            next_state, _, _ = model.observe_step(self.state, actions, torch.from_numpy(np.stack(next_obs)),
                torch.zeros(4, dtype=torch.bool), stochastic=False)
            # Capture the same-episode final feature/value BEFORE constructing reset worlds.
            next_values = self.agent.target_value(model.features(next_state)).squeeze(-1).clone()
            row = {'features': features, 'actions': actions, 'rewards': torch.tensor(rewards),
                   'terminated': torch.tensor(terminated), 'truncated': torch.tensor(truncated),
                   'next_values': next_values}
            for key, value in row.items():data[key].append(value)
            for i in range(4):
                if terminated[i] or truncated[i]:
                    self.worlds[i] = self._world()
                    reset = first_state(self.agent, self.worlds[i].observe())
                    next_state.deter[i].copy_(reset.deter[0]); next_state.stoch[i].copy_(reset.stoch[0])
            self.state = next_state
            self.steps += 4
        return {key: torch.stack(values) for key, values in data.items()}

    def update(self, batch):
        features = batch['features'].flatten(0, 1)
        logits, value = self.agent.predict(features)
        with torch.no_grad():
            returns = survival_targets(batch['rewards'], batch['terminated'], batch['truncated'],
                batch['next_values'], torch.ones_like(batch['terminated']), gamma=.99).flatten()
        advantage = returns-value.detach()
        advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
        logprob = logits.log_softmax(-1)
        actor = -(logprob.gather(1, batch['actions'].flatten()[:, None]).squeeze(1)*advantage).mean()
        critic = F.mse_loss(value, returns)
        entropy = -(logprob.exp()*logprob).sum(-1).mean()
        loss = actor+.5*critic-.01*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real-policy loss')
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.agent.trainable_parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step(); self.agent.update_target(.99); self.updates += 1
        return {'real_steps': self.steps, 'updates': self.updates, 'actor_loss': float(actor.detach()),
                'critic_loss': float(critic.detach()), 'entropy': float(entropy.detach()), 'gradient_norm': float(norm)}


def run(output):
    output = Path(output)
    if output.exists():raise ValueError('Choose a fresh output directory')
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(PROTOCOL, indent=2)+'\n')
    torch.set_num_threads(1); torch.manual_seed(0)
    start = time.perf_counter()
    replay, data = collect_diverse(episodes=384, seed_base=30000)
    world = WorldModel(replay.spec, ModelConfig(deter=32, hidden=32, stoch=4, classes=4))
    learner = WorldModelTrainer(world, learning_rate=.001)
    sampler = np.random.default_rng(60000)
    for _ in range(300):learner.step(fixed_target_batch(replay, sampler))
    torch.manual_seed(123)
    agent = PosteriorActorCritic(world)
    original_world = {k: v.clone() for k, v in world.state_dict().items()}
    untrained = deepcopy(agent).eval()
    trainer = OnlinePosteriorTrainer(agent)
    candidates, log = [], []
    for _ in range(16384//64):
        log.append(trainer.update(trainer.collect()))
        if trainer.steps in PROTOCOL['checkpoints']:
            validation, _ = evaluate(agent, PROTOCOL['validation_seeds'])
            candidates.append({'training_steps': trainer.steps, 'updates': trainer.updates,
                'validation': validation, 'weights': deepcopy(agent.state_dict())})
            print(json.dumps({'phase': 'trained_checkpoint', 'steps': trainer.steps,
                              'validation_return': validation['mean_alive_step_return']}), flush=True)
    best = max(candidates, key=lambda c: (c['validation']['mean_alive_step_return'], -c['training_steps']))
    agent.load_state_dict(best['weights'])
    for key, value in original_world.items():
        torch.testing.assert_close(value, agent.world_model.state_dict()[key], rtol=0, atol=0)
    checkpoint = {'format': 'posterior-control-v1', 'protocol': PROTOCOL, 'model_config': asdict(world.config),
                  'model': agent.state_dict(), 'training_steps': best['training_steps'], 'updates': best['updates'],
                  'world_model_pretraining_updates': 300, 'world_model_replay_sha256': replay_digest(replay)}
    checkpoint_io.save(checkpoint, output/'selected-policy.safe')
    checkpoint_hash = hashlib.sha256((output/'selected-policy.safe').read_bytes()).hexdigest()
    frozen = {'checkpoint_sha256': checkpoint_hash, 'selected_training_steps': best['training_steps'],
              'selected_validation': best['validation'], 'final_seeds': PROTOCOL['final_seeds'],
              'recording_seed': 70000, 'final_set_role': 'report_only_no_selection'}
    (output/'selection-frozen.json').write_text(json.dumps(frozen, indent=2)+'\n')
    comparisons = []
    for method in ['trained', 'untrained', 'random', 'greedy']:
        result, _ = evaluate(untrained if method == 'untrained' else agent, PROTOCOL['final_seeds'], method=method)
        comparisons.append(result)
    recorded, frames = evaluate(agent, [70000], record=True)
    # Resume parity checks the actual deterministic policy rollout, not just model losses.
    restored = PosteriorActorCritic(WorldModel(world.spec, world.config))
    restored.load_state_dict(checkpoint_io.load(output/'selected-policy.safe')['model'])
    repeated, repeated_frames = evaluate(restored, [70000], record=True)
    if frames != repeated_frames or recorded['episodes'] != repeated['episodes']:
        raise AssertionError('Trained rollout differs after safe checkpoint reload')
    source = {'world_model': hashlib.sha256(Path(world_module.__file__).read_bytes()).hexdigest(),
              'control': hashlib.sha256(Path(control_module.__file__).read_bytes()).hexdigest(),
              'adapter': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report = {'protocol': PROTOCOL, 'selected_training_steps': best['training_steps'], 'selected_updates': best['updates'],
        'total_policy_real_steps': trainer.steps, 'total_policy_updates': trainer.updates,
        'world_model_real_steps': data['real_steps'], 'world_model_updates': 300,
        'world_model_frozen_during_policy_training': True, 'checkpoint_sha256': checkpoint_hash,
        'parameters': sum(p.numel() for p in agent.parameters()),
        'trainable_policy_value_parameters': sum(p.numel() for p in agent.trainable_parameters()),
        'source_sha256': source, 'validation_candidates': [{k: v for k, v in c.items() if k != 'weights'} for c in candidates],
        'final_comparisons': comparisons, 'representative_recording': recorded,
        'checkpoint_recording_resume_match': True, 'elapsed_seconds': time.perf_counter()-start,
        'trained_policy': True, 'complete_hybrid': False, 'planning': False,
        'synthetic_mechanics_implemented': False, 'cost_usd': 0,
        'limitations': ['One policy training seed; no continual-learning/retention result',
                       'RSSM posterior categorical modes; learned prior not used for planning',
                       'Linear actor/critic attachment is a minimal prototype, not complete Draft 2',
                       'Legacy resource environment; synthetic hazards and wear remain planned']}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'recording.json').write_text(json.dumps({'config': asdict(CONFIG), 'frames': frames[0]}, separators=(',', ':'))+'\n')
    (output/'updates.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in log))
    print(json.dumps({'phase': 'complete', 'selected_steps': best['training_steps'], 'final_scores':
                     {r['method']: r['mean_alive_step_return'] for r in comparisons}, 'recording_return': recorded['mean_alive_step_return'],
                     'seconds': report['elapsed_seconds']}), flush=True)
    return report
