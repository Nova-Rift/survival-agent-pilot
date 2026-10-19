"""Fixed-protocol CPU diagnostic. No tuning on final seeds, control or new mechanics."""
from copy import deepcopy
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from survival_env.simulator import Config
from survival_agent.hybrid import world_model, diagnostics
from survival_agent.hybrid.world_model import WorldModel, ModelConfig, WorldModelTrainer
from survival_agent.hybrid.diagnostics import episode_predictions, prediction_scores
from survival_adapters.hybrid_survival import collect_replay


PROTOCOL = {
    'version': 1, 'training_seeds': list(range(32)),
    'development_seeds': list(range(1000, 1016)), 'final_seeds': list(range(2000, 2016)),
    'model_seeds': [0, 1, 2], 'updates_per_condition': 80, 'batch_size': 4,
    'burn_in': 1, 'prefix_unroll': 3, 'diagnostic_context': 1,
    'conditions': ['prefix', 'complete_episode'],
    'environment': asdict(Config(size=6, agents=1, radius=1, max_hunger=6,
                                initial_food=6, food_per_step=1, horizon=8)),
    'model': asdict(ModelConfig(deter=32, hidden=32, stoch=4, classes=4)),
    'learning_rate': .001, 'maximum_optimizer_updates': 480,
    'selection': 'Fixed before final-seed evaluation; report every condition/seed, no final-set tuning',
    'comparison_limit': 'Equal optimizer updates, unequal supervised transitions and compute; diagnostic, not matched-compute causal comparison',
}


def episode_row(episode, *, burn_in=1):
    row = deepcopy(episode)
    length = len(row['actions'])
    if not 0 <= burn_in < length:
        raise ValueError('Invalid complete-episode burn-in')
    row.update(loss_mask=np.arange(length) >= burn_in,
               context_is_episode_start=True, episode_offset=0)
    return row


def sample_complete_episodes(replay, rng, *, batch_size=4, burn_in=1):
    """Sample a length group weighted by episode count, then complete equal-length rows.

    No joining, zero-padding, interior resets or missing endpoint observations.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('Invalid batch size')
    eligible = [e for e in replay.episodes if len(e['actions']) > burn_in]
    if not eligible:
        raise ValueError('No complete episodes fit burn-in')
    length = len(eligible[int(rng.integers(len(eligible)))]['actions'])
    group = [e for e in eligible if len(e['actions']) == length]
    return [episode_row(group[int(rng.integers(len(group)))], burn_in=burn_in) for _ in range(batch_size)]


def coverage(rows):
    selected = [(row, np.asarray(row['loss_mask'])) for row in rows]
    return {'supervised_transitions': sum(int(mask.sum()) for _, mask in selected),
            'death_labels': sum(int(row['terminated'][mask].sum()) for row, mask in selected),
            'continuing_labels': sum(int((~row['terminated'][mask]).sum()) for row, mask in selected),
            'exclusive_truncation_labels': sum(int((row['truncated'][mask] & ~row['terminated'][mask]).sum()) for row, mask in selected)}


def prefix_rows(replay):
    rows = []
    for episode in replay.episodes:
        if len(episode['actions']) < 4:
            continue
        row = {k: None if v is None else v[:4+(k in ['observations', 'is_first', 'masks'])].copy() for k, v in episode.items()}
        row.update(loss_mask=np.array([False, True, True, True]), context_is_episode_start=True, episode_offset=0)
        rows.append(row)
    return rows


def score_set(model, replay, baseline):
    records = []
    for episode in replay.episodes:
        # In this unchanged one-agent survival task, reward is the alive indicator.
        np.testing.assert_array_equal(episode['rewards'], (~episode['terminated']).astype(np.float32))
        records.extend(episode_predictions(model, episode_row(episode), context=1))
    horizons = sorted({r['horizon'] for r in records})
    def score(subset):
        return {'model': prediction_scores(subset),
                'training_mean_baseline': prediction_scores(subset, constant=baseline),
                'always_alive_baseline': prediction_scores(subset, constant=(1., 1.))}
    return {'all': score(records),
            'endpoint_only': score([r for r in records if r['terminated'] or r['truncated']]),
            'by_horizon': {str(h): score([r for r in records if r['horizon'] == h]) for h in horizons}}


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh diagnostic output directory')
    output.mkdir(parents=True)
    # Persist the complete fixed plan BEFORE collecting or evaluating final seeds.
    (output/'protocol.json').write_text(json.dumps(PROTOCOL, indent=2)+'\n')
    torch.set_num_threads(1)
    start = time.perf_counter()
    config = Config(**PROTOCOL['environment'])
    training, train_report = collect_replay(episodes=32, seed=0, config=config)
    development, dev_report = collect_replay(episodes=16, seed=1000, config=config)
    full_rows = [episode_row(e) for e in training.episodes]
    labels = np.concatenate([e['rewards'] for e in training.episodes])
    continuations = np.concatenate([~e['terminated'] for e in training.episodes])
    baseline = (float(labels.mean()), float(continuations.mean()))
    # Reproduce the original smoke's dataset to audit what its prefix windows expose.
    original, original_report = collect_replay(episodes=16, seed=0)
    audit = {'original_collection': original_report,
             'original_prefix_unique_episode_coverage': coverage(prefix_rows(original)),
             'original_complete_episode_coverage': coverage([episode_row(e) for e in original.episodes]),
             'broader_prefix_unique_episode_coverage': coverage(prefix_rows(training)),
             'broader_complete_episode_coverage': coverage(full_rows)}
    models, logs = [], []
    for seed in PROTOCOL['model_seeds']:
        for condition in PROTOCOL['conditions']:
            torch.manual_seed(seed)
            model = WorldModel(training.spec, ModelConfig(**PROTOCOL['model']))
            trainer = WorldModelTrainer(model, learning_rate=PROTOCOL['learning_rate'])
            # Independent repeatable row stream for this condition, not held-out data.
            replay_rng = np.random.default_rng(seed+3000)
            training.rng = np.random.default_rng(seed+3000)
            totals = {key: 0 for key in coverage(full_rows)}
            run_start = time.perf_counter()
            for update in range(PROTOCOL['updates_per_condition']):
                rows = training.sample(4, burn_in=1, unroll=3) if condition == 'prefix' else sample_complete_episodes(training, replay_rng)
                counts = coverage(rows)
                for key in totals:
                    totals[key] += counts[key]
                metrics = trainer.step(rows)
                logs.append({'model_seed': seed, 'condition': condition, 'update': update+1, **counts, **metrics})
            models.append((model, {'model_seed': seed, 'condition': condition, 'updates': trainer.updates,
                                  'target_counts': totals, 'training_seconds': time.perf_counter()-run_start}))
    # All optimization is complete before final data are collected or scored.
    final, final_report = collect_replay(episodes=16, seed=2000, config=config)
    split_seeds = [set(r['seed'] for r in report['episodes']) for report in [train_report, dev_report, final_report]]
    if any(split_seeds[i] & split_seeds[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError('Episode seeds must be disjoint')
    for report in [train_report, dev_report, final_report]:
        if not any(r['terminated'] for r in report['episodes']) or not any(r['truncated'] and not r['terminated'] for r in report['episodes']):
            raise AssertionError('Fixed diagnostic split lacks required death/truncation coverage; do not retune final seeds')
    runs = []
    for model, metadata in models:
        runs.append({**metadata, 'development': score_set(model, development, baseline),
                     'final': score_set(model, final, baseline)})
    initial = []
    for seed in PROTOCOL['model_seeds']:
        torch.manual_seed(seed)
        model = WorldModel(training.spec, ModelConfig(**PROTOCOL['model']))
        initial.append({'model_seed': seed, 'final': score_set(model, final, baseline)})
    report = {'protocol': PROTOCOL, 'device': 'cpu', 'threads': 1,
              'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__},
              'source_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                for name, module in [('world_model', world_model), ('diagnostics', diagnostics)]},
              'datasets': {'training': train_report, 'development': dev_report, 'final': final_report},
              'coverage_audit': audit, 'training_mean_baseline': baseline,
              'runs': runs, 'initial_models': initial,
              'optimizer_updates': sum(r['updates'] for _, r in models),
              'elapsed_seconds': time.perf_counter()-start, 'cost_usd': 0,
              'survival_improvement_claim': False, 'continual_learning_claim': False,
              'final_seeds_used_for_optimization_or_selection': False}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'updates.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in logs))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({'output_files': ['protocol.json', 'report.json', 'updates.jsonl'],
                      'optimizer_updates': report['optimizer_updates'], 'elapsed_seconds': report['elapsed_seconds']}))
