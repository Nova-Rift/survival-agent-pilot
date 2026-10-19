"""Fixed-budget 7k-transition data-coverage experiment; CPU, no learned control."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from survival_agent.hybrid import world_model, diagnostics
from survival_agent.hybrid.world_model import WorldModel, ModelConfig, WorldModelTrainer
from survival_agent.hybrid.diagnostics import episode_predictions, prediction_scores
from survival_adapters import coverage_data
from survival_adapters.coverage_data import (collect_diverse, first_episodes, fixed_target_batch,
                                              replay_digest, CONFIGURATIONS, POLICIES)
from survival_adapters.hybrid_survival import collect_replay
from examples.hybrid_stage_b_diagnostic import episode_row, coverage


PROTOCOL = {
    'version': 1, 'training_seed_base': 30000, 'training_episodes': 384,
    'development_seed_base': 40000, 'development_episodes': 64,
    'final_seed_base': 50000, 'final_episodes': 128,
    'historical_training_seeds': list(range(32)), 'small_diverse_training_episodes': 16,
    'configurations': [asdict(c) for c in CONFIGURATIONS], 'policies': list(POLICIES),
    'configuration_policy_sampling': 'Independent uniform pre-outcome draws, no outcome filtering/resampling',
    'model_seeds': [0, 1, 2], 'conditions': ['historical_249', 'diverse_small', 'diverse_full'],
    'updates_per_condition': 300, 'batch_size': 4, 'unroll': 4,
    'supervised_transitions_per_condition': 4800, 'maximum_optimizer_updates': 2700,
    'burn_in': 'Exact actual episode prefix, length varies; gradients stopped before last four transitions',
    'window_sampling': 'Episode-count-weighted length group; 50% endpoint, 50% uniform shared end index',
    'learning_rate': .001, 'gradient_clip': 10.,
    'model': asdict(ModelConfig(deter=32, hidden=32, stoch=4, classes=4)),
    'diagnostic_context': 1, 'diagnostic_latent': 'Categorical mode; not outcome averaging',
    'selection': 'No architecture/objective/hyperparameter selection; all predeclared conditions retained',
    'final_rule': 'Freeze hashes after development evaluation; collect final episodes only afterward; no final-set tuning',
    'comparability': 'Same optimizer steps and loss-target count, variable burn-in CPU cost; varied-small is a size control with only 16 episodes, not identical coverage',
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def data_summary(replay):
    term = np.concatenate([e['terminated'] for e in replay.episodes])
    trunc = np.concatenate([e['truncated'] for e in replay.episodes])
    rewards = np.concatenate([e['rewards'] for e in replay.episodes])
    np.testing.assert_array_equal(rewards, (~term).astype(np.float32))
    return {'episodes': len(replay.episodes), 'real_transitions': len(term),
            'deaths': int(term.sum()), 'exclusive_truncations': int((trunc & ~term).sum()),
            'both': int((term & trunc).sum()), 'continuing_transitions': int((~term).sum()),
            'replay_sha256': replay_digest(replay),
            'training_mean_baseline': (float(rewards.mean()), float((~term).mean()))}


def score_records(records, baseline):
    return {'model': prediction_scores(records),
            'training_mean_baseline': prediction_scores(records, constant=baseline),
            'always_alive_baseline': prediction_scores(records, constant=(1., 1.))}


def score_set(model, replay, baseline):
    records = []
    for episode in replay.episodes:
        records.extend(episode_predictions(model, episode_row(episode, burn_in=0), context=1))
    # Natural episode/prevalence mixture; no balancing, filtering or replacements.
    return {'all': score_records(records, baseline),
            'endpoint_only': score_records([r for r in records if r['terminated'] or r['truncated']], baseline),
            'by_horizon': {str(h): score_records([r for r in records if r['horizon'] == h], baseline)
                           for h in sorted({r['horizon'] for r in records})}}


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh diagnostic output directory')
    output.mkdir(parents=True)
    write_json(output/'planned_protocol.json', PROTOCOL)
    torch.set_num_threads(1)
    start = time.perf_counter()
    diverse, diverse_collection = collect_diverse(episodes=384, seed_base=30000)
    development, development_collection = collect_diverse(episodes=64, seed_base=40000)
    historical, historical_collection = collect_replay(episodes=32, seed=0,
        config=__import__('survival_env.simulator', fromlist=['Config']).Config(size=6, agents=1, radius=1,
            max_hunger=6, initial_food=6, food_per_step=1, horizon=8))
    datasets = {'historical_249': historical, 'diverse_small': first_episodes(diverse, 16), 'diverse_full': diverse}
    summaries = {name: data_summary(data) for name, data in datasets.items()}
    if not 5000 <= summaries['diverse_full']['real_transitions'] <= 10000:
        raise AssertionError('Fixed training collection missed target; no final data have been collected')
    print(json.dumps({'phase': 'training_collection', 'datasets': summaries}), flush=True)
    models, logs, initial_hashes = [], [], {}
    for seed in PROTOCOL['model_seeds']:
        for name, dataset in datasets.items():
            torch.manual_seed(seed)
            model = WorldModel(dataset.spec, ModelConfig(**PROTOCOL['model']))
            initial_hash = model_digest(model)
            if seed in initial_hashes and initial_hashes[seed] != initial_hash:
                raise AssertionError('Conditions must start from identical parameters')
            initial_hashes[seed] = initial_hash
            trainer = WorldModelTrainer(model, learning_rate=.001)
            sampler = np.random.default_rng(seed+60000)
            target_counts = {k: 0 for k in coverage([episode_row(dataset.episodes[0])])}
            context_transitions = 0
            run_start = time.perf_counter()
            for update in range(PROTOCOL['updates_per_condition']):
                rows = fixed_target_batch(dataset, sampler, batch_size=4, unroll=4)
                counts = coverage(rows)
                for key in counts:
                    target_counts[key] += counts[key]
                context_transitions += sum(len(row['actions']) for row in rows)
                loss = trainer.step(rows)
                logs.append({'model_seed': seed, 'condition': name, 'update': update+1, **counts, **loss})
            if target_counts['supervised_transitions'] != 4800:
                raise AssertionError('Conditions must receive identical supervised-transition budgets')
            models.append((model, {'model_seed': seed, 'condition': name, 'updates': trainer.updates,
                'target_counts': target_counts, 'processed_context_transitions': context_transitions,
                'training_seconds': time.perf_counter()-run_start,
                'initial_model_sha256': initial_hash, 'trained_model_sha256': model_digest(model)}))
            print(json.dumps({'phase': 'trained', 'seed': seed, 'condition': name, 'target_counts': target_counts}), flush=True)
    runs = []
    for model, metadata in models:
        baseline = tuple(summaries[metadata['condition']]['training_mean_baseline'])
        runs.append({**metadata, 'development': score_set(model, development, baseline)})
    frozen = {'protocol': PROTOCOL, 'datasets': summaries,
              'model_versions': [{k: run[k] for k in ['model_seed', 'condition', 'initial_model_sha256', 'trained_model_sha256']} for run in runs],
              'selected_changes': [], 'selection_note': 'All planned conditions retained. No model/objective/hyperparameter tuning after development.',
              'final_data_collected': False}
    write_json(output/'frozen_protocol.json', frozen)
    frozen_hash = hashlib.sha256((output/'frozen_protocol.json').read_bytes()).hexdigest()
    print(json.dumps({'phase': 'protocol_frozen_before_final', 'sha256': frozen_hash}), flush=True)
    final, final_collection = collect_diverse(episodes=128, seed_base=50000)
    for (model, metadata), report in zip(models, runs):
        baseline = tuple(summaries[metadata['condition']]['training_mean_baseline'])
        report['final'] = score_set(model, final, baseline)
    seed_sets = [set(e['seed'] for e in c['episodes']) for c in
                 [historical_collection, diverse_collection, development_collection, final_collection]]
    if any(seed_sets[i] & seed_sets[j] for i in range(4) for j in range(i+1, 4)):
        raise AssertionError('Training/evaluation map seeds overlap')
    report = {'protocol': PROTOCOL, 'frozen_protocol_sha256': frozen_hash,
              'source_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                for name, module in [('world_model', world_model), ('diagnostics', diagnostics), ('coverage_data', coverage_data)]},
              'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__}, 'device': 'cpu', 'threads': 1,
              'training_datasets': summaries,
              'collection': {'historical': historical_collection, 'diverse': diverse_collection,
                             'development': development_collection, 'final': final_collection},
              'development_summary': data_summary(development), 'final_summary': data_summary(final),
              'runs': runs, 'optimizer_updates': sum(r['updates'] for r in runs),
              'elapsed_seconds': time.perf_counter()-start, 'cost_usd': 0,
              'final_seeds_used_for_optimization_or_selection': False,
              'trained_policy': False, 'survival_improvement_claim': False, 'continual_learning_claim': False}
    write_json(output/'report.json', report)
    (output/'updates.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in logs))
    print(json.dumps({'phase': 'complete', 'optimizer_updates': report['optimizer_updates'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
