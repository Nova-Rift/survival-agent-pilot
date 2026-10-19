from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
from survival_agent import checkpoint_io
from survival_agent.hybrid.transformer_agent import TransformerWorldModel, TransformerAgent
from survival_adapters.synthetic_control import Bridge, SPEC, SyntheticTrainer, observe, policy_actions
from survival_adapters.transformer_control import MODEL, load_agent, run_smoke

ENV = Path(__file__).resolve().parents[2]/'survival-env/browser'


class TransformerControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_transformer_collect_matches_independent_environment_replay_and_actions(self):
        agent = TransformerAgent(TransformerWorldModel(SPEC, MODEL))
        bridge, other = Bridge(ENV), Bridge(ENV)
        try:
            trainer = SyntheticTrainer(bridge, agent); trainer.next_seed = 150000
            batch = trainer.collect()
            result = other.reset(150000); state = observe(agent.world_model, result)
            for t in range(16):
                torch.testing.assert_close(batch['features'][t], agent.world_model.features(state), rtol=0, atol=0)
                rng1, rng2 = torch.Generator().manual_seed(t), torch.Generator().manual_seed(t)
                logits, _ = agent.predict(agent.world_model.features(state))
                torch.testing.assert_close(agent.act(state, generator=rng1),
                    policy_actions(logits, action_mode='sampled', generator=rng2), rtol=0, atol=0)
                result = other.step(batch['actions'][t].numpy())
                self.assertEqual(batch['rewards'][t].tolist(), result['rewards'])
                state = observe(agent.world_model, result, state, batch['actions'][t])
            torch.testing.assert_close(state.history, trainer.state.history, rtol=0, atol=0)
            saved = deepcopy(trainer.state)
            other.reset(150001)
            fresh = observe(agent.world_model, other.reset(150001))
            self.assertEqual(fresh.lengths.tolist(), [1]*8)
            torch.testing.assert_close(trainer.state.history, saved.history, rtol=0, atol=0)
        finally:
            bridge.close(); other.close()

    def test_training_checkpoint_reload_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'smoke'
            report = run_smoke(output, ENV)
            self.assertTrue(report['attention_weights_changed'])
            self.assertTrue(report['actor_weights_changed'])
            self.assertTrue(report['world_frozen_during_policy_updates'])
            self.assertTrue(report['checkpoint_reload_frames_match'])
            self.assertEqual(report['policy_joint_steps'], 32)
            self.assertEqual(len(report['world_updates']), 8)
            with self.assertRaises(FileExistsError):run_smoke(output, ENV)
            state = checkpoint_io.load(output/'transformer-smoke.safe')
            state['model_config']['context'] = 32
            checkpoint_io.save(state, Path(tmp)/'wrong-config.safe')
            with self.assertRaises(ValueError):load_agent(Path(tmp)/'wrong-config.safe')
            checkpoint_io.save({'format': 'synthetic-posterior-control-v1'}, Path(tmp)/'gru.safe')
            with self.assertRaises(ValueError):load_agent(Path(tmp)/'gru.safe')


if __name__ == '__main__':
    unittest.main()
