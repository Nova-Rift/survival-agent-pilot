import unittest
from pathlib import Path
import numpy as np
import torch
from survival_agent.hybrid.world_model import WorldModel
from survival_agent.hybrid.control import PosteriorActorCritic
from survival_adapters.synthetic_control import Bridge, SyntheticTrainer, SPEC, MODEL, observe, policy_actions, evaluate
from survival_adapters.training import survival_targets

ENV = Path(__file__).resolve().parents[2]/'survival-env/browser'

class SyntheticControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_eight_channel_environment_and_reset_reproducibility(self):
        bridge=Bridge(ENV)
        try:
            initial=bridge.reset(0)
            self.assertEqual(np.asarray(initial['observations']).shape,(8,200))
            self.assertEqual(initial['config']['wear_limit'],96)
            self.assertEqual(initial['config']['agents'],8)
            result=initial;frames=[]
            while not result['done']:
                result=bridge.step(scripted=True);frames.append(result['frame'])
            self.assertEqual(result['frame']['tick'],96)
            self.assertFalse(any(result['frame']['alive']))
            bridge.reset(0);again=[]
            while True:
                result=bridge.step(scripted=True);again.append(result['frame'])
                if result['done']:break
            self.assertEqual(frames,again)
        finally:bridge.close()

    def test_inactive_mask_and_death_do_not_bootstrap_or_leak_reset(self):
        rewards=torch.tensor([[1.,0.],[0.,0.],[1.,0.]])
        terminated=torch.tensor([[False,False],[True,False],[False,False]])
        active=torch.tensor([[True,False],[True,False],[True,False]])
        values=torch.full((3,2),99.)
        result=survival_targets(rewards,terminated,torch.zeros_like(active),values,active,gamma=1.)
        self.assertEqual(result[:,1].tolist(),[0.,0.,0.])
        self.assertEqual(result[:2,0].tolist(),[1.,0.])

    def test_real_update_changes_heads_preserves_world_and_evaluation_isolation(self):
        torch.manual_seed(0)
        bridge=Bridge(ENV);other=Bridge(ENV)
        try:
            agent=PosteriorActorCritic(WorldModel(SPEC,MODEL))
            world={k:v.clone() for k,v in agent.world_model.state_dict().items()}
            actor=agent.actor.weight.clone()
            trainer=SyntheticTrainer(bridge,agent)
            batch=trainer.collect()
            self.assertEqual(batch['features'].shape,(16,8,48))
            self.assertGreater(int(batch['active'].sum()),0)
            metrics=trainer.update(batch)
            self.assertTrue(np.isfinite(list(metrics.values())).all())
            self.assertFalse(torch.equal(actor,agent.actor.weight))
            for k,v in world.items():torch.testing.assert_close(v,agent.world_model.state_dict()[k],rtol=0,atol=0)
            state=bridge.step([0]*8)
            other.reset(220000);other.step(scripted=True)
            next_state=bridge.step([0]*8)
            self.assertEqual(next_state['frame']['tick'],state['frame']['tick']+1)
        finally:bridge.close();other.close()

    def test_sampled_policy_matches_training_distribution_and_has_fixed_rng(self):
        logits=torch.zeros(4096,5)
        actual=policy_actions(logits,action_mode='sampled',generator=torch.Generator().manual_seed(123))
        expected=torch.multinomial(logits.softmax(-1),1,generator=torch.Generator().manual_seed(123)).squeeze(-1)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(set(actual.tolist()),set(range(5)))
        self.assertEqual(set(policy_actions(logits,action_mode='greedy').tolist()),{0})
        with self.assertRaises(ValueError):policy_actions(logits,action_mode='sampled')

    def test_real_final_truncation_value_cuts_reset_recursion(self):
        rewards=torch.tensor([[1.],[1000.]])
        truncated=torch.tensor([[True],[False]])
        targets=survival_targets(rewards,torch.zeros_like(truncated),truncated,
            torch.tensor([[7.],[9999.]]),torch.ones_like(truncated),gamma=.5)
        self.assertEqual(float(targets[0,0]),4.5)

    def test_collect_observation_alignment_matches_independent_same_action_replay(self):
        torch.manual_seed(0)
        agent=PosteriorActorCritic(WorldModel(SPEC,MODEL));bridge=Bridge(ENV);other=Bridge(ENV)
        try:
            trainer=SyntheticTrainer(bridge,agent);batch=trainer.collect()
            result=other.reset(110000);state=observe(agent.world_model,result)
            for t in range(16):
                torch.testing.assert_close(batch['features'][t],agent.world_model.features(state),rtol=0,atol=0)
                result=other.step(batch['actions'][t].numpy())
                self.assertEqual(batch['rewards'][t].tolist(),result['rewards'])
                self.assertEqual(batch['terminated'][t].tolist(),result['info']['died'])
                state=observe(agent.world_model,result,state,batch['actions'][t])
            torch.testing.assert_close(agent.world_model.features(trainer.state),agent.world_model.features(state),rtol=0,atol=0)
        finally:bridge.close();other.close()

    def test_sampled_evaluation_reproducible_and_does_not_change_global_rng(self):
        torch.manual_seed(0);agent=PosteriorActorCritic(WorldModel(SPEC,MODEL));bridge=Bridge(ENV)
        try:
            rng=torch.get_rng_state().clone()
            first,frames=evaluate(bridge,agent,[440000],record=True,action_mode='sampled')
            second,repeated=evaluate(bridge,agent,[440000],record=True,action_mode='sampled')
            self.assertEqual(first,second);self.assertEqual(frames,repeated)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            self.assertGreater(sum(x>0 for x in first['episodes'][0]['action_counts']),1)
        finally:bridge.close()

if __name__=='__main__':unittest.main()
