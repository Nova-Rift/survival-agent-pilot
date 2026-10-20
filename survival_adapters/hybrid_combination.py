"""Bounded single-task CPU integration check for the experimental hybrid.

Real Gymnasium collection is confined to this adapter. This is an engineering
check, not a benchmark, checkpoint selection protocol or survival experiment.
"""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
import gymnasium as gym
from survival_agent.hybrid import model, policy, learning, planning, runtime, checkpoints, world_model
from survival_agent.hybrid.model import HybridModel, HybridConfig, TransformerConfig
from survival_agent.hybrid.learning import HybridLearner, LearningConfig, replay_states, planner_target
from survival_agent.hybrid.runtime import HybridController
from survival_agent.hybrid.planning import PlannerConfig, mcts, mpc
from survival_agent.hybrid.contracts import EnvSpec, VectorAdapter
from survival_agent.hybrid.replay import SequenceReplay


def sources():
    modules=(model,policy,learning,planning,runtime,world_model)
    paths=[Path(inspect.getfile(module)) for module in modules]+[Path(__file__)]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(output, task='CartPole-v1', *, distillation=False):
    if task not in ('CartPole-v1','Pendulum-v1'):
        raise ValueError('This bounded check supports CartPole or Pendulum')
    output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.set_num_threads(1);torch.manual_seed(20261019)
    rng=np.random.default_rng(20261019)
    env=gym.make(task,max_episode_steps=16)
    adapter=VectorAdapter(env)
    replay=SequenceReplay(adapter.spec,seed=20261019)
    transformer=TransformerConfig(deter=16,hidden=16,stoch=2,classes=3,context=8)
    head=HybridConfig(hidden=16,q_ensemble=3)
    learn=LearningConfig(horizon=3,distillation_weight=.1 if distillation else 0.)
    plan=PlannerConfig(horizon=3,simulations=8,candidates=16,actor_candidates=2,elites=4,iterations=2)
    protocol={'purpose':'bounded CPU engineering check; no learning-improvement claim',
              'task':task,'seed':20261019,'collection_episodes':4,'episode_limit':16,
              'update_cycles':4,'evaluation_episodes_per_mode':1,'evaluation_seed':20261020,
              'transformer':asdict(transformer),'heads':asdict(head),'learning':asdict(learn),
              'planner':asdict(plan),'source_sha256':sources(),'device':'cpu','paid_cost_usd':0}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.perf_counter();transitions=0
    try:
        for episode in range(4):
            obs,_=adapter.reset(seed=20261019+episode)
            replay.start(obs)
            for _ in range(16):
                action=int(rng.integers(adapter.spec.action.count)) if adapter.spec.action.kind=='discrete' else rng.uniform(-1,1,len(adapter.spec.action.low)).astype(np.float32)
                item=adapter.step(action);transitions+=1
                replay.append(action,item.reward,item.next_observation,item.terminated,item.truncated)
                if item.terminated or item.truncated:break
        network=HybridModel(adapter.spec,transformer,head)
        learner=HybridLearner(network,learn)
        logs=[]
        for update in range(4):
            rows=replay.sample(1,burn_in=update,unroll=1)
            metrics={'world':learner.world_step(rows)}
            target=None
            if distillation:
                state,_,_=replay_states(network,rows)
                result=(mcts if adapter.spec.action.kind=='discrete' else mpc)(network,state,plan)
                target=planner_target(rows,result,version=learner.model_version,step=learner.updates,settings=asdict(plan))
            metrics['imagination']=learner.imagination_step(rows,target)
            if adapter.spec.action.kind=='continuous':metrics['td']=learner.td_step(rows)
            logs.append(metrics)
        checkpoints.save(output/'hybrid.safe',replay,components=learner.state_dict(),config=protocol,rng=rng,counters={'updates':learner.updates})
        # Evaluate the same frozen weights with actor and relevant planner, without
        # claiming one short episode establishes a performance comparison.
        returns={};actions={}
        for mode in ('actor','auto'):
            controller=HybridController(network,mode=mode,config=plan)
            obs,_=adapter.reset(seed=20261020)
            total=0.;sequence=[]
            for _ in range(16):
                result=controller.decide(obs)
                action=int(result.action.item()) if adapter.spec.action.kind=='discrete' else result.action[0].cpu().numpy()
                sequence.append(np.asarray(action).tolist())
                item=adapter.step(action);total+=item.reward;obs=item.next_observation
                controller.boundary(terminated=item.terminated,truncated=item.truncated)
                if item.terminated or item.truncated:break
            returns[controller.mode]=total;actions[controller.mode]=sequence
        restored=HybridLearner(HybridModel(adapter.spec,transformer,head),learn)
        _,saved,_=checkpoints.load(output/'hybrid.safe',adapter.spec,config=protocol,rng=rng)
        restored.load_state_dict(saved)
        for key,value in network.state_dict().items():
            torch.testing.assert_close(value,restored.model.state_dict()[key],rtol=0,atol=0)
        report={'protocol':protocol,'real_training_transitions':transitions,'updates':logs,
                'frozen_checkpoint_single_episode_returns':returns,'evaluation_actions':actions,
                'checkpoint_weights_match':True,'elapsed_seconds':time.perf_counter()-started,
                'limitations':['No benchmark or survival-improvement result','No continual-learning or retention test',
                               'Planning uses deterministic latent modes','Public website and social game are separate']}
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for path in output.iterdir():path.chmod(0o600)
        return report
    finally:env.close()
