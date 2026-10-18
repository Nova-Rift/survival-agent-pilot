import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from simulator import SurvivalWorld, random_policy, greedy_policy
from agent import HistoryAgent, history_start, history_append, smoke_update, save_checkpoint, load_checkpoint, make_agent_population

torch.set_num_threads(1)

def transition(world,rng,history):
    obs=world.observe();actions=random_policy(obs,rng)
    next_obs,reward,_,_=world.step(actions)
    next_history=history_append(history,next_obs)
    batch={'history':torch.from_numpy(history),'actions':torch.from_numpy(actions),
           'next_history':torch.from_numpy(next_history),'rewards':torch.from_numpy(reward),
           'alive':torch.from_numpy(world.alive.astype(np.float32)),
           'teacher_actions':torch.from_numpy(greedy_policy(obs,rng))}
    return batch,next_history

class AgentTests(unittest.TestCase):
    def test_equal_initial_models_and_explicit_weight_modes(self):
        torch.manual_seed(7);template=HistoryAgent()
        shared=make_agent_population(template,2,weight_mode='shared')
        copies=make_agent_population(template,2,weight_mode='independent_copies')
        self.assertIs(shared[0],shared[1]);self.assertIsNot(copies[0],copies[1])
        for k,v in template.state_dict().items():
            for model in copies:torch.testing.assert_close(model.state_dict()[k],v,atol=0,rtol=0)
        with torch.no_grad():next(copies[0].parameters()).add_(1)
        self.assertFalse(torch.equal(next(copies[0].parameters()),next(copies[1].parameters())))
        for k,v in template.state_dict().items():torch.testing.assert_close(copies[1].state_dict()[k],v,atol=0,rtol=0)
        with self.assertRaises(ValueError):make_agent_population(template,2,weight_mode='automatic')

    def test_per_agent_history_does_not_mix_between_batch_rows(self):
        model=HistoryAgent().eval()
        obs=np.zeros((2,5,5,5),dtype=np.float32);history=history_start(obs)
        with torch.no_grad():before=model.represent(torch.from_numpy(history)).clone()
        obs[0,0,2,2]=1
        changed=history_append(history,obs)
        np.testing.assert_array_equal(changed[1],history[1])
        self.assertEqual(history[0].sum(),0) # Appending did not mutate retained history.
        with torch.no_grad():after=model.represent(torch.from_numpy(changed))
        torch.testing.assert_close(before[1],after[1],rtol=0,atol=0)

    def test_shared_model_shapes_and_planning_determinism(self):
        torch.manual_seed(4);model=HistoryAgent().eval();world=SurvivalWorld(3)
        h=torch.from_numpy(history_start(world.observe()))
        with torch.no_grad():
            z=model.represent(h);p,v=model.predict(z)
            self.assertEqual(tuple(p.shape),(8,5));self.assertEqual(tuple(v.shape),(8,))
            torch.testing.assert_close(p.softmax(-1).sum(-1),torch.ones(8))
            a,q=model.plan(h);b,r=model.plan(h)
            torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(q,r,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(q).all());self.assertTrue(((a>=0)&(a<5)).all())

    def test_checkpoint_resume_reproduces_next_update_and_world(self):
        torch.manual_seed(9);m=HistoryAgent();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
        world=SurvivalWorld(8);rng=np.random.default_rng(6);h=history_start(world.observe())
        batch,h=transition(world,rng,h);smoke_update(m,opt,batch)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.safe';save_checkpoint(path,m,opt,world,rng,h,1)
            batch,h=transition(world,rng,h);losses=smoke_update(m,opt,batch)
            expected={k:v.clone() for k,v in m.state_dict().items()};frame=world.frame()
            restored,ropt,rworld,rrng,rh,updates=load_checkpoint(path)
            self.assertEqual(updates,1)
            rbatch,rh=transition(rworld,rrng,rh);rlosses=smoke_update(restored,ropt,rbatch)
            self.assertEqual(frame,rworld.frame());self.assertEqual(losses,rlosses)
            np.testing.assert_array_equal(h,rh)
            for k,v in restored.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertTrue(path.exists());self.assertFalse(path.with_suffix('.tmp').exists())

if __name__=='__main__':unittest.main()
