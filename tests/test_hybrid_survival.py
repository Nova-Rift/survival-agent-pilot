from pathlib import Path
import tempfile
import unittest
import numpy as np
from survival_env.simulator import Config
from survival_agent.hybrid.contracts import VectorAdapter
from survival_adapters.hybrid_survival import SurvivalVectorEnv, collect_replay
from examples.hybrid_stage_b import run


class HybridSurvivalTests(unittest.TestCase):
    def test_single_agent_final_death_observation_and_explicit_reset(self):
        env = SurvivalVectorEnv(Config(size=4, agents=1, radius=1, max_hunger=1,
                                       initial_food=0, food_per_step=0, horizon=3))
        adapter = VectorAdapter(env, env.spec)
        initial, _ = adapter.reset(seed=1)
        self.assertTrue(initial.any())
        transition = adapter.step(0)
        self.assertTrue(transition.terminated); self.assertFalse(transition.truncated)
        self.assertFalse(transition.next_observation.any())
        with self.assertRaises(RuntimeError):adapter.step(0)
        reset, _ = adapter.reset(seed=2)
        self.assertTrue(reset.any())
        self.assertFalse(transition.next_observation.any(), 'reset must not overwrite saved final observation')

    def test_horizon_retains_final_observation_and_continuation(self):
        env = SurvivalVectorEnv(Config(size=4, agents=1, radius=1, max_hunger=5,
                                       initial_food=0, food_per_step=0, horizon=1))
        adapter = VectorAdapter(env, env.spec); adapter.reset(seed=3)
        transition = adapter.step(0)
        self.assertFalse(transition.terminated); self.assertTrue(transition.truncated)
        self.assertTrue(transition.next_observation.any()); self.assertEqual(transition.continuation, 1.)
        with self.assertRaises(ValueError):SurvivalVectorEnv(Config(agents=2))

    def test_real_replay_has_separate_complete_episodes_and_local_vectors(self):
        replay, report = collect_replay(episodes=4, seed=0)
        self.assertEqual(len(replay.episodes), 4)
        self.assertEqual(replay.spec.observation_width, 45)
        self.assertEqual(report['real_steps'], sum(len(e['actions']) for e in replay.episodes))
        for episode in replay.episodes:
            replay.validate_episode(episode)
            self.assertTrue(episode['terminated'][-1] or episode['truncated'][-1])
            self.assertEqual(episode['observations'].dtype, np.float32)

    def test_bounded_real_world_training_and_safe_resume_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            report = run(output, updates=2)
            self.assertEqual(report['optimizer_updates'], 2)
            self.assertTrue(report['checkpoint_diagnostics_match'])
            self.assertFalse(report['survival_improvement_claim'])
            self.assertFalse(report['trained_policy'])
            self.assertTrue((output/'world-model.safe').is_file())
            self.assertEqual(len((output/'updates.jsonl').read_text().splitlines()), 2)
            with self.assertRaises(ValueError):run(output, updates=2)
            with self.assertRaises(ValueError):run(Path(tmp)/'invalid', updates=1000)


if __name__ == '__main__':
    unittest.main()
