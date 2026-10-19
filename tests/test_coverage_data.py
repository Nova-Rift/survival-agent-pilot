from copy import deepcopy
import unittest
import numpy as np
from survival_agent.hybrid.world_model import sequence_batch
from survival_adapters.coverage_data import (collect_diverse, first_episodes, fixed_target_batch, replay_digest)
from examples.hybrid_coverage_diagnostic import PROTOCOL


class CoverageDataTests(unittest.TestCase):
    def test_repeatable_collection_preserves_all_real_episode_outcomes(self):
        a, report_a = collect_diverse(episodes=8, seed_base=30000)
        b, report_b = collect_diverse(episodes=8, seed_base=30000)
        self.assertEqual(replay_digest(a), replay_digest(b))
        self.assertEqual(report_a, report_b)
        self.assertEqual(len(a.episodes), 8)
        self.assertEqual(report_a['deaths']+report_a['exclusive_truncations'], 8)
        for episode in a.episodes:
            a.validate_episode(episode)
            np.testing.assert_array_equal(episode['rewards'], (~episode['terminated']).astype(np.float32))
        with self.assertRaises(ValueError):collect_diverse(episodes=385, seed_base=0)

    def test_matched_loss_count_exact_prefix_and_endpoints_without_mutation(self):
        replay, _ = collect_diverse(episodes=8, seed_base=30000)
        before = replay_digest(replay)
        saw_endpoint = False
        saw_context = False
        for seed in range(12):
            rows = fixed_target_batch(replay, np.random.default_rng(seed))
            batch = sequence_batch(rows, replay.spec)
            self.assertEqual(int(batch['loss_mask'].sum()), 16)
            self.assertEqual(len({len(r['actions']) for r in rows}), 1)
            for row in rows:
                self.assertTrue(row['is_first'][0]); self.assertFalse(row['is_first'][1:].any())
                self.assertTrue(row['loss_mask'][-1]); self.assertEqual(int(row['loss_mask'].sum()), 4)
                saw_endpoint |= bool(row['terminated'][-1] or row['truncated'][-1])
                saw_context |= len(row['actions']) > 4
                row['observations'][:] = -99
        self.assertTrue(saw_endpoint); self.assertTrue(saw_context)
        self.assertEqual(replay_digest(replay), before)
        subset = first_episodes(replay, 2)
        subset.episodes[0]['observations'][:] = 100
        self.assertEqual(replay_digest(replay), before)

    def test_fixed_protocol_budgets_and_new_disjoint_final_seeds(self):
        self.assertEqual(PROTOCOL['updates_per_condition']*PROTOCOL['batch_size']*PROTOCOL['unroll'], 4800)
        self.assertEqual(len(PROTOCOL['conditions'])*len(PROTOCOL['model_seeds'])*PROTOCOL['updates_per_condition'], 2700)
        train = set(range(PROTOCOL['training_seed_base'], PROTOCOL['training_seed_base']+PROTOCOL['training_episodes']))
        dev = set(range(PROTOCOL['development_seed_base'], PROTOCOL['development_seed_base']+PROTOCOL['development_episodes']))
        final = set(range(PROTOCOL['final_seed_base'], PROTOCOL['final_seed_base']+PROTOCOL['final_episodes']))
        self.assertFalse(train & dev or train & final or dev & final)
        self.assertFalse(final & set(range(2000, 2016)), 'old final set remains historical')


if __name__ == '__main__':
    unittest.main()
