from survival_agent import checkpoint_io
"""Search math, replay boundaries and real tiny encoder/unroll/resume tests."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from survival_agent.muzero import MuZeroModel,MuZeroLearner,Episode,EpisodeReplay,mcts,value_target,collect_episode
from tests.test_integrated import synthetic_saved_fixture

torch.set_num_threads(1)

class ToyModel:
    def __init__(self,terminal=True):self.calls=0;self.terminal=terminal
    def initial(self,x):return torch.zeros(1,1),torch.zeros(1,5),torch.tensor([2.])
    def recurrent(self,z,actions):
        self.calls+=1
        reward=(actions==0).float()
        return z,reward,torch.zeros(1,5),torch.tensor([2.]),torch.tensor([20. if self.terminal else -20.])

def one_step_episode(terminal=True):
    return Episode(np.zeros((2,4,5,5,5),np.float32),np.array([0]),np.array([0. if terminal else 1.],np.float32),
                   np.array([[1,0,0,0,0]],np.float32),np.array([0.,4.],np.float32),
                   np.array([terminal]),np.array([not terminal]))

class SearchMathTests(unittest.TestCase):
    def test_visit_normalization_reward_backup_and_terminal_no_expansion(self):
        model=ToyModel();result=mcts(model,torch.zeros(1,1),simulations=32,gamma=.5)
        self.assertEqual(int(result['visits'].sum()),32)
        self.assertAlmostEqual(float(result['policy'].sum()),1.)
        self.assertEqual(int(result['visits'].argmax()),0)
        self.assertAlmostEqual(result['value'],result['visits'][0]/32)
        self.assertLessEqual(model.calls,5) # Predicted terminal leaves never expand further.
        self.assertTrue(all(not child.children for child in result['root'].children.values()))
        with patch.object(model,'initial',side_effect=AssertionError('Actual terminal root must not query model')):
            dead=mcts(model,torch.zeros(1,1),root_terminal=True)
        self.assertEqual(dead['value'],0);self.assertEqual(dead['policy'].sum(),0)

    def test_nonterminal_discounted_backup(self):
        result=mcts(ToyModel(terminal=False),torch.zeros(1,1),simulations=1,max_depth=1,gamma=.5)
        self.assertEqual(result['value'],2.) # Immediate 1 + gamma * leaf value 2.

class ReplayTests(unittest.TestCase):
    def test_terminal_truncation_and_padding_never_cross_episode(self):
        terminal=one_step_episode();truncated=one_step_episode(False)
        self.assertEqual(value_target(terminal,0,3,.5),0)
        self.assertEqual(value_target(truncated,0,3,.5),3) # Reward 1 + bootstrap .5*4.
        replay=EpisodeReplay(0);replay.add(terminal)
        sample=replay.sample(1,unroll=4)
        torch.testing.assert_close(sample['transition_mask'],torch.tensor([[1.,0,0,0]]))
        torch.testing.assert_close(sample['policy_mask'],torch.tensor([[1.,0,0,0,0]]))
        torch.testing.assert_close(sample['value_mask'],torch.tensor([[1.,1.,0,0,0]]))
        self.assertEqual(float(sample['terminal'][0,0]),1.)
        invalid=deepcopy(terminal);invalid.terminated[0]=False
        with self.assertRaises(ValueError):replay.add(invalid)
        invalid=deepcopy(terminal);invalid.policies[:]=0
        with self.assertRaises(ValueError):replay.add(invalid)

class RealMuZeroPlumbingTests(unittest.TestCase):
    def test_scratch_encoder_dynamics_heads_update_and_exact_replay_resume(self):
        model=MuZeroModel(seed=7);learner=MuZeroLearner(model,seed=8)
        episodes,info=collect_episode(model,seed=1,max_steps=3,simulations=4)
        self.assertTrue(info['complete']);self.assertEqual(len(episodes),8)
        for episode in episodes:learner.replay.add(episode)
        before={k:v.clone() for k,v in model.state_dict().items()}
        loss=learner.update(batch_size=2,unroll=2)
        self.assertTrue(all(np.isfinite(v) for v in loss.values()))
        for group in ['core.embed','core.history_encoder','core.transition','core.policy','core.value','core.reward','terminal']:
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(group)),group)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.safe';checkpoint_io.save(learner.state(),path)
            expected_loss=learner.update();expected={k:v.clone() for k,v in model.state_dict().items()}
            resumed=MuZeroLearner(MuZeroModel(seed=7),seed=999);resumed.load_state(checkpoint_io.load(path,weights_only=True))
            self.assertEqual(expected_loss,resumed.update())
            for k,v in resumed.model.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertEqual(resumed.updates,2)

    def test_real_tiny_qwen_frozen_feature_option_and_explicit_finetune_limit(self):
        with synthetic_saved_fixture('qwen2') as backend:
            model=MuZeroModel(encoder_mode='frozen_language_features',feature_width=backend.feature_width,encoder_provenance=backend.provenance)
            before={k:v.clone() for k,v in backend.model.state_dict().items()}
            episodes,info=collect_episode(model,seed=0,max_steps=1,simulations=2,language_backend=backend)
            learner=MuZeroLearner(model)
            for episode in episodes:learner.replay.add(episode)
            self.assertTrue(np.isfinite(learner.update(unroll=1)['total']))
            for k,v in backend.model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for p in backend.model.parameters()))
        with self.assertRaises(NotImplementedError):MuZeroModel(language_encoder_training='fine_tunable')

if __name__=='__main__':unittest.main()
