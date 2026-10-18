import checkpoint_io
"""Trainer correctness, not a scientific training experiment."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
import torch
from language_backend import LocalLanguageBackend,BackendConfig
from test_integrated import FeatureMockBackend
from training import Trainer,TrainConfig,survival_targets,evaluate

torch.set_num_threads(1)

class TargetTests(unittest.TestCase):
    def test_death_vs_truncation_and_no_reset_episode_leakage(self):
        rewards=torch.tensor([[1.],[1.],[1.]])
        terminated=torch.zeros((3,1),dtype=torch.bool);truncated=terminated.clone();truncated[1]=True
        values=torch.tensor([[10.],[20.],[30.]])
        active=torch.ones((3,1),dtype=torch.bool)
        target=survival_targets(rewards,terminated,truncated,values,active,gamma=.5)
        torch.testing.assert_close(target,torch.tensor([[6.5],[11.],[16.]]))
        terminated[1]=True;truncated[1]=False;rewards[1]=0
        target=survival_targets(rewards,terminated,truncated,values,active,gamma=.5)
        self.assertEqual(float(target[1]),0);self.assertEqual(float(target[0]),1)
        active[2]=False
        self.assertEqual(float(survival_targets(rewards,terminated,truncated,values,active,.5)[2]),0)

class RealScratchTrainerTests(unittest.TestCase):
    def test_finite_updates_all_head_groups_and_exact_resume(self):
        backend=LocalLanguageBackend(BackendConfig(mode='from_scratch',seed=4,max_new_tokens=2))
        config=TrainConfig(train_seeds=(5,),init_seed=7,horizon=4,rollout_steps=1)
        trainer=Trainer(backend,config)
        before={k:v.clone() for k,v in trainer.controller.core.state_dict().items()}
        batch=trainer.collect(1);loss=trainer.update(batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in loss.values()))
        for group in ['project','transition','reward','value','policy']:
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in trainer.controller.core.state_dict().items() if k.startswith(group)))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in backend.model.parameters()))
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'checkpoint.safe';trainer.save(file)
            next_batch=trainer.collect(1);expected_loss=trainer.update(next_batch)
            expected={k:v.clone() for k,v in trainer.controller.core.state_dict().items()}
            resumed=Trainer(backend,config);resumed.load_state(checkpoint_io.load(file,weights_only=True))
            actual_batch=resumed.collect(1)
            for k,v in next_batch.items():torch.testing.assert_close(v,actual_batch[k],rtol=0,atol=0)
            self.assertEqual(expected_loss,resumed.update(actual_batch))
            self.assertEqual(trainer.world.frame(),resumed.world.frame())
            for k,v in resumed.controller.core.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertEqual(resumed.updates,2)
            bad=deepcopy(resumed.state());bad['backend']['seed']=999
            with self.assertRaises(ValueError):resumed.load_state(bad)

class MockTrainerBoundaryTests(unittest.TestCase):
    def test_collector_marks_real_death_and_time_limit_separately(self):
        death=Trainer(FeatureMockBackend(),TrainConfig(rollout_steps=1))
        death.world.food[:]=False;death.world.hunger[:]=1
        death.world.config=replace(death.world.config,food_per_step=0)
        batch=death.collect(1)
        self.assertTrue(batch['terminated'].all());self.assertFalse(batch['truncated'].any())
        self.assertEqual(float(batch['rewards'].sum()),0)
        limit=Trainer(FeatureMockBackend(),TrainConfig(horizon=1,rollout_steps=1))
        batch=limit.collect(1)
        self.assertFalse(batch['terminated'].any());self.assertTrue(batch['truncated'].all())
        self.assertEqual(float(batch['rewards'].sum()),8)

    def test_collect_limits_and_inactive_slots_cannot_affect_loss(self):
        a=Trainer(FeatureMockBackend(),TrainConfig(rollout_steps=2,horizon=3))
        tick=a.world.tick
        self.assertIsNone(a.collect(1,time.monotonic()-1));self.assertEqual(a.world.tick,tick)
        self.assertIsNone(a.collect(0));self.assertEqual(a.world.tick,tick)
        batch=a.collect(1);batch['active'][:,7]=False
        changed=deepcopy(batch);changed['rewards'][:,7]=10000
        b=Trainer(FeatureMockBackend(),a.config)
        self.assertEqual(a.update(batch),b.update(changed))

    def test_heldout_all_four_comparisons_and_partial_scores(self):
        trainer=Trainer(FeatureMockBackend(),TrainConfig(horizon=2,rollout_steps=1))
        trainer.update(trainer.collect(1));state=trainer.state()
        with self.assertRaises(ValueError):evaluate(trainer.backend,state,seeds=(0,),horizon=2)
        report=evaluate(trainer.backend,state,seeds=(1001,),horizon=2,max_joint_steps=8,max_seconds=5)
        self.assertTrue(report['comparisons_complete'])
        self.assertEqual(set(report['means']),{'random','greedy','no_planning','planning'})
        self.assertTrue(all(r['seed']==1001 and r['complete'] for r in report['episodes']))
        partial=evaluate(trainer.backend,state,seeds=(1001,),horizon=2,max_joint_steps=1,max_seconds=5)
        self.assertFalse(partial['comparisons_complete']);self.assertIsNone(partial['means'])

if __name__=='__main__':unittest.main()
