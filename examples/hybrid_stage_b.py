"""Bounded CPU-only RSSM training smoke on real single-agent survival replay.

Random/greedy collection; no learned policy, planner or survival comparison.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from survival_agent.hybrid import checkpoints
from survival_agent.hybrid import world_model
from survival_agent.hybrid.world_model import (WorldModel, ModelConfig, WorldModelTrainer,
                                             sequence_loss, prediction_diagnostics)
from survival_adapters.hybrid_survival import collect_replay


def run(output, *, seed=0, updates=40):
    if type(seed) is not int or not 0 <= seed < 2**32-2000 or type(updates) is not int or not 1 <= updates <= 100:
        raise ValueError('Seed/update budget outside CPU smoke bounds')
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    start = time.perf_counter()
    replay, collected = collect_replay(episodes=16, seed=seed)
    heldout, heldout_collection = collect_replay(episodes=4, seed=seed+1000)
    model = WorldModel(replay.spec, ModelConfig(deter=32, hidden=32, stoch=4, classes=4))
    trainer = WorldModelTrainer(model, learning_rate=.001)
    fixed_train = replay.sample(4, burn_in=1, unroll=3)
    fixed_heldout = heldout.sample(4, burn_in=1, unroll=3)
    @torch.no_grad()
    def evaluate():
        return {'train_mode_loss': {k: float(v) for k, v in sequence_loss(model, fixed_train, stochastic=False).items()},
                'heldout_mode_loss': {k: float(v) for k, v in sequence_loss(model, fixed_heldout, stochastic=False).items()},
                'heldout_prior_errors': prediction_diagnostics(model, fixed_heldout, context=1, horizons=(1, 3))}
    before = evaluate()
    log = []
    for update in range(updates):
        metrics = trainer.step(replay.sample(4, burn_in=1, unroll=3))
        log.append({'update': update+1, **metrics})
    after = evaluate()
    rng = np.random.default_rng(seed)
    config = {'stage': 'B', 'model': asdict(model.config), 'environment': collected['config'],
              'learning_rate': .001, 'burn_in': 1, 'unroll': 3, 'batch_size': 4,
              'model_source_sha256': hashlib.sha256(Path(world_model.__file__).read_bytes()).hexdigest()}
    path = output/'world-model.safe'
    checkpoints.save(path, replay, components={'model': model.state_dict(), 'optimizer': trainer.optimizer.state_dict()},
                     config=config, rng=rng, counters={'optimizer_updates': trainer.updates, 'real_steps': collected['real_steps']})
    restored_model = WorldModel(replay.spec, model.config)
    restored_trainer = WorldModelTrainer(restored_model, learning_rate=.001)
    restored, components, counters = checkpoints.load(path, replay.spec, config=config, rng=rng)
    restored_model.load_state_dict(components['model'])
    restored_trainer.optimizer.load_state_dict(components['optimizer'])
    restored_trainer.updates = counters['optimizer_updates']
    resumed_diagnostics = prediction_diagnostics(restored_model, fixed_heldout, context=1, horizons=(1, 3))
    if resumed_diagnostics != after['heldout_prior_errors']:
        raise AssertionError('Fixed held-out diagnostics differ after safe checkpoint resume')
    report = {'stage': 'B recurrent world-model CPU smoke', 'seed': seed,
              'runtime': checkpoints.versions(), 'device': 'cpu', 'threads': 1,
              'config': config, 'parameters': sum(p.numel() for p in model.parameters()),
              'training_collection': collected, 'heldout_collection': heldout_collection,
              'optimizer_updates': trainer.updates, 'before': before, 'after': after,
              'first_update': log[0], 'last_update': log[-1],
              'checkpoint_diagnostics_match': True, 'checkpoint_replay_episodes': len(restored.episodes),
              'elapsed_seconds': time.perf_counter()-start, 'cost_usd': 0,
              'trained_policy': False, 'planners_implemented': False, 'survival_improvement_claim': False,
              'limitations': ['Single training seed and tiny fixed held-out episode subset',
                              'Categorical-mode diagnostics approximate stochastic outcomes',
                              'Training updates only world-model prediction losses',
                              'No actor/value learning or survival/retention evaluation']}
    (output/'updates.jsonl').write_text(''.join(json.dumps(item, allow_nan=False)+'\n' for item in log))
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int, default=40)
    args = parser.parse_args()
    print(json.dumps(run(args.output, seed=args.seed, updates=args.updates), indent=2))
