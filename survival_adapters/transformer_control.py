"""Small CPU engineering check of the transformer agent on real browser rules.

This verifies collection, optimization and checkpoint replay, not useful control.
Historical GRU protocols and final evaluation seeds are left intact.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from survival_agent import checkpoint_io
from survival_agent.hybrid import transformer_agent, world_model
from survival_agent.hybrid.transformer_agent import TransformerConfig, TransformerWorldModel, TransformerAgent
from survival_agent.hybrid.world_model import WorldModelTrainer
from survival_agent.hybrid.replay import SequenceReplay
from . import synthetic_control, training
from .synthetic_control import Bridge, SPEC, SyntheticTrainer, evaluate
from .coverage_data import fixed_target_batch, replay_digest

MODEL = TransformerConfig()
PROTOCOL = {
    'id': 'transformer-agent-smoke-001', 'purpose': 'engineering verification only',
    'architecture': MODEL.architecture, 'model_config': asdict(MODEL),
    'environment_version': 'synthetic-survival-v2.0',
    'prefit_seeds': [1010000, 1010001], 'policy_training_seed_start': 150000,
    'replay_check_seed': 1020000, 'world_updates': 8, 'policy_updates': 2,
    'joint_steps_per_update': 16, 'model_seed': 0, 'policy_seed': 123,
    'action_mode': 'sampled', 'cpu_only': True, 'cost_usd': 0,
    'final_evaluation': False, 'checkpoint_selection': 'last smoke update; no performance selection',
    'claim': 'No survival improvement or continual-learning claim',
}


def sources():
    modules = [transformer_agent, world_model, synthetic_control, training]
    paths = [Path(inspect.getfile(m)) for m in modules] + [Path(__file__),
        Path(__file__).resolve().parents[1]/'scripts/synthetic-bridge.mjs']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_agent(path):
    state = checkpoint_io.load(path)
    keys = {'format', 'spec', 'model_config', 'source_sha256', 'weights', 'world_updates', 'policy_updates'}
    if not isinstance(state, dict) or set(state) != keys or state['format'] != 'transformer-posterior-control-v1':
        raise ValueError('Expected a transformer checkpoint; GRU weights are incompatible')
    if state['spec'] != asdict(SPEC) or state['model_config'] != asdict(MODEL) or state['source_sha256'] != sources():
        raise ValueError('Transformer checkpoint specification/configuration/source mismatch')
    agent = TransformerAgent(TransformerWorldModel(SPEC, MODEL))
    agent.load_state_dict(state['weights'], strict=True)
    return agent


def collect_smoke_replay(bridge):
    replay = SequenceReplay(SPEC, capacity=4096, max_bytes=4*1024*1024)
    transitions = 0
    for episode, seed in enumerate(PROTOCOL['prefit_seeds']):
        rng = np.random.default_rng(seed)
        result = bridge.reset(seed)
        streams = [SequenceReplay(SPEC, capacity=128) for _ in range(8)]
        for i, stream in enumerate(streams):
            stream.start(result['observations'][i])
        while not result['done']:
            previous = result
            result = bridge.step(rng.integers(5, size=8), scripted=episode == 1)
            for i, stream in enumerate(streams):
                if previous['frame']['alive'][i]:
                    stream.append(result['actions'][i], result['rewards'][i], result['observations'][i],
                                  bool(result['info']['died'][i]), bool(result['info']['truncated']))
                    transitions += 1
        for stream in streams:
            assert stream.pending is None and len(stream.episodes) == 1
            replay.episodes.extend(stream.episodes)
    assert transitions <= replay.capacity
    assert sum(replay.nbytes(e) for e in replay.episodes) <= replay.max_bytes
    return replay, transitions


def run_smoke(output, environment):
    output = Path(output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    environment = Path(environment).resolve()
    protocol = {**PROTOCOL, 'source_sha256': sources(), 'environment_source_sha256':
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in environment.glob('*.mjs')}}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    torch.set_num_threads(1); torch.manual_seed(PROTOCOL['model_seed'])
    started = time.perf_counter()
    bridge = Bridge(environment)
    try:
        replay, transitions = collect_smoke_replay(bridge)
        model = TransformerWorldModel(SPEC, MODEL)
        initial_attention = model.core.transformer.layers[0].self_attn.in_proj_weight.detach().clone()
        world_trainer = WorldModelTrainer(model, learning_rate=.001)
        rng = np.random.default_rng(1010010)
        world_logs = [world_trainer.step(fixed_target_batch(replay, rng))
                      for _ in range(PROTOCOL['world_updates'])]
        assert not torch.equal(initial_attention, model.core.transformer.layers[0].self_attn.in_proj_weight)
        torch.manual_seed(PROTOCOL['policy_seed'])
        agent = TransformerAgent(model)
        frozen = deepcopy(model.state_dict())
        initial_actor = agent.actor.weight.detach().clone()
        trainer = SyntheticTrainer(bridge, agent)
        trainer.next_seed = PROTOCOL['policy_training_seed_start']
        policy_logs = [trainer.update(trainer.collect()) for _ in range(PROTOCOL['policy_updates'])]
        assert not torch.equal(initial_actor, agent.actor.weight)
        for key, value in frozen.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        checkpoint = output/'transformer-smoke.safe'
        checkpoint_io.save({'format': 'transformer-posterior-control-v1', 'spec': asdict(SPEC),
            'model_config': asdict(MODEL), 'source_sha256': sources(), 'weights': agent.state_dict(),
            'world_updates': world_trainer.updates, 'policy_updates': trainer.updates}, checkpoint)
        original, frames = evaluate(bridge, agent, [PROTOCOL['replay_check_seed']], record=True, action_mode='sampled')
        restored = load_agent(checkpoint)
        repeated, replayed = evaluate(bridge, restored, [PROTOCOL['replay_check_seed']], record=True, action_mode='sampled')
        assert original == repeated and frames == replayed
        report = {'protocol': protocol, 'world_training_transitions': transitions,
            'replay_sha256': replay_digest(replay), 'world_updates': world_logs, 'policy_updates': policy_logs,
            'policy_joint_steps': trainer.joint_steps, 'policy_active_steps': trainer.active_steps,
            'attention_weights_changed': True, 'actor_weights_changed': True,
            'world_frozen_during_policy_updates': True, 'checkpoint_reload_frames_match': True,
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'parameters': sum(p.numel() for p in agent.parameters()),
            'elapsed_seconds': time.perf_counter()-started, 'cost_usd': 0,
            'limitations': ['Engineering smoke only; no held-out performance comparison',
                'No language input, continual-learning mechanism, retention test or planning',
                'Weights-only checkpoint; not a mid-episode optimizer/environment resume']}
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        for path in output.iterdir():
            path.chmod(0o600)
        print(json.dumps({k: report[k] for k in ['parameters', 'policy_joint_steps',
            'checkpoint_reload_frames_match', 'elapsed_seconds', 'cost_usd']}), flush=True)
        return report
    finally:
        bridge.close()
