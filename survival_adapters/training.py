from survival_agent import checkpoint_io
"""Minimal bounded on-policy actor-critic + one-step latent-model trainer.
NOT full MuZero: no MCTS targets, replay training, recurrent unroll or encoder tuning.
The encoder is always frozen, even with the pretrained config. CPU-only entrypoints.
"""
import argparse
import survival_agent
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.nn import functional as F
from survival_env.simulator import Config,SurvivalWorld,random_policy,greedy_policy
from survival_agent.integrated import IntegratedController
from survival_agent.language_backend import LocalLanguageBackend,read_config

@dataclass(frozen=True)
class TrainConfig:
    train_seeds: tuple = (0,1,2)
    init_seed: int = 0
    horizon: int = 64
    rollout_steps: int = 4
    learning_rate: float = 0.001
    gamma: float = 0.99
    target_ema: float = 0.99

    def __post_init__(self):
        if not self.train_seeds or len(set(self.train_seeds))!=len(self.train_seeds) or any(type(s) is not int or s<0 for s in self.train_seeds):raise ValueError('Distinct nonnegative training seeds required')
        if not 1<=self.horizon<=128 or not 1<=self.rollout_steps<=32:raise ValueError('Bounded horizon/rollout required')
        if self.learning_rate<=0 or not 0<self.gamma<=1 or not 0<=self.target_ema<1:raise ValueError('Invalid learning parameters')

def source_hashes():
    import inspect
    sources={'training.py':Path(__file__), 'integrated.py':Path(inspect.getfile(IntegratedController)),
             'language_backend.py':Path(inspect.getfile(LocalLanguageBackend)),
             'simulator.py':Path(inspect.getfile(SurvivalWorld))}
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}


def runtime_versions():return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}

def survival_targets(rewards,terminated,truncated,next_values,active,gamma=.99):
    """N-step returns. Death => zero bootstrap. Time limit => bootstrap from the
    true final observation, then cut recursion so a reset episode cannot leak in.
    A rollout cutoff also bootstraps. Inactive-agent slots never contribute.
    """
    target=torch.zeros_like(rewards);running=next_values[-1]
    for t in reversed(range(len(rewards))):
        boundary=terminated[t]|truncated[t]
        continuation=next_values[t] if t==len(rewards)-1 else torch.where(boundary,next_values[t],running)
        running=(rewards[t]+gamma*torch.where(terminated[t],torch.zeros_like(continuation),continuation))*active[t]
        target[t]=running
    return target

class Trainer:
    def __init__(self,backend,config=TrainConfig()):
        self.backend=backend;self.config=config
        self.controller=IntegratedController(backend,seed=config.init_seed,planning=False)
        self.target=deepcopy(self.controller.core).eval().requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.controller.core.parameters(),lr=config.learning_rate)
        self.action_rng=torch.Generator().manual_seed(config.init_seed)
        self.episode_index=0;self.joint_steps=0;self.updates=0;self.world=None
        self._reset()

    def _reset(self):
        seed=self.config.train_seeds[self.episode_index%len(self.config.train_seeds)]
        self.world=SurvivalWorld(seed,replace(Config(),horizon=self.config.horizon))
        self.controller.reset_all()

    def collect(self,limit,deadline=float('inf')):
        fields={k:[] for k in ['features','next_features','actions','rewards','active','terminated','truncated','observation','next_observation']}
        for _ in range(min(limit,self.config.rollout_steps)):
            if time.monotonic()>=deadline:break # Cooperative boundary: finish an atomic joint step.
            if self.world.done:self.episode_index+=1;self._reset()
            obs=self.world.observe();active=self.world.alive.copy()
            features=torch.zeros((8,self.backend.feature_width));actions=torch.zeros(8,dtype=torch.long)
            for i in np.flatnonzero(active):
                event,f,_=self.controller.encode_local(int(i),obs[i]);features[i]=f[0]
                with torch.no_grad():
                    logits,_=self.controller.core.predict(self.controller.core.represent(f))
                    actions[i]=torch.multinomial(logits.softmax(-1)[0],1,generator=self.action_rng)[0]
                self.controller.record_decision(int(i),event,int(actions[i]))
            next_obs,reward,_,info=self.world.step(actions.numpy())
            next_features=torch.zeros_like(features)
            for i in np.flatnonzero(self.world.alive):
                _,f,_=self.controller.encode_local(int(i),next_obs[i]);next_features[i]=f[0]
            row={'features':features,'next_features':next_features,'actions':actions,
                 'rewards':torch.from_numpy(reward),'active':torch.from_numpy(active),
                 'terminated':torch.from_numpy(info['died']),
                 'truncated':torch.full((8,),info['truncated'],dtype=torch.bool)&torch.from_numpy(self.world.alive),
                 'observation':torch.from_numpy(obs),'next_observation':torch.from_numpy(next_obs)}
            for name,value in row.items():fields[name].append(value)
            self.joint_steps+=1
        return {k:torch.stack(v) for k,v in fields.items()} if fields['features'] else None

    def update(self,batch):
        core=self.controller.core;core.train()
        t,n,d=batch['features'].shape
        z=core.represent(batch['features'].reshape(-1,d));policy,value=core.predict(z)
        child,reward=core.dynamics(z,batch['actions'].flatten())
        with torch.no_grad():
            target_z=self.target.represent(batch['next_features'].reshape(-1,d))
            _,next_values=self.target.predict(target_z)
            returns=survival_targets(batch['rewards'],batch['terminated'],batch['truncated'],next_values.reshape(t,n),batch['active'],self.config.gamma).flatten()
        mask=batch['active'].flatten().float();denom=mask.sum().clamp_min(1)
        model_mask=mask*(~batch['terminated'].flatten()).float()
        advantage=(returns-value).detach()
        logprob=policy.log_softmax(-1).gather(1,batch['actions'].flatten()[:,None]).squeeze(1)
        losses={'policy_actor_critic':(-logprob*advantage*mask).sum()/denom,
                'value':((value-returns).square()*mask).sum()/denom,
                'reward':((reward-batch['rewards'].flatten()).square()*mask).sum()/denom,
                'dynamics':((child-target_z).square().mean(-1)*model_mask).sum()/model_mask.sum().clamp_min(1)}
        loss=losses['policy_actor_critic']+.5*losses['value']+losses['reward']+losses['dynamics']
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        self.optimizer.zero_grad();loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in core.parameters()):raise RuntimeError('Non-finite gradient')
        norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.)
        self.optimizer.step()
        with torch.no_grad():
            for target,param in zip(self.target.parameters(),core.parameters()):
                target.mul_(self.config.target_ema).add_(param,alpha=1-self.config.target_ema)
        self.updates+=1;self.controller.training_updates=self.updates;core.eval()
        return {k:float(v.detach()) for k,v in losses.items()}|{'total':float(loss.detach()),'gradient_norm_before_clip':float(norm)}

    def state(self):
        return deepcopy({'format':1,'algorithm':'on_policy_actor_critic_plus_one_step_latent_model',
            'config':asdict(self.config),'backend':self.backend.provenance,'source_hashes':source_hashes(),
            'versions':runtime_versions(),
            'controller':self.controller.state(),'target':self.target.state_dict(),'optimizer':self.optimizer.state_dict(),
            'world':self.world.state(),'action_rng':self.action_rng.get_state(),'torch_rng':torch.get_rng_state(),
            'episode_index':self.episode_index,'joint_steps':self.joint_steps,'updates':self.updates})

    def load_state(self,state):
        if state['format']!=1 or state['config']!=asdict(self.config) or state['backend']!=self.backend.provenance or state['source_hashes']!=source_hashes() or state['versions']!=runtime_versions():raise ValueError('Checkpoint config/encoder/code/runtime provenance mismatch')
        self.controller.load_state(state['controller']);self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer']);self.world.load_state(state['world'])
        self.action_rng.set_state(state['action_rng']);torch.set_rng_state(state['torch_rng'])
        self.episode_index=state['episode_index'];self.joint_steps=state['joint_steps'];self.updates=state['updates']

    def save(self,path):
        path=Path(path);temporary=path.with_suffix('.tmp')
        checkpoint_io.save(self.state(),temporary);os.replace(temporary,path)

def evaluate(backend,state,seeds=(1001,1002),horizon=32,max_joint_steps=256,max_seconds=30):
    if not seeds or set(seeds)&set(state['config']['train_seeds']) or len(set(seeds))!=len(seeds):raise ValueError('Evaluation seeds must be distinct and held out from training')
    if any(type(s) is not int or s<0 for s in seeds):raise ValueError('Invalid evaluation seeds')
    if state['backend']!=backend.provenance or state['source_hashes']!=source_hashes():raise ValueError('Evaluation encoder/code provenance mismatch')
    if not 1<=horizon<=128 or max_joint_steps<1 or max_seconds<=0:raise ValueError('Invalid evaluation limit')
    deadline=time.monotonic()+max_seconds;rows=[];used=0
    for method in ['random','greedy','no_planning','planning']:
        for seed in seeds:
            world=SurvivalWorld(seed,replace(Config(),horizon=horizon));rng=np.random.default_rng(seed+1000)
            controller=None
            if method in ['no_planning','planning']:
                controller=IntegratedController(backend,seed=state['config']['init_seed'],planning=method=='planning')
                controller.core.load_state_dict(state['controller']['core']);controller.training_updates=state['updates']
            reward_sum=0.;food=0;steps=0
            while not world.done:
                if used>=max_joint_steps or time.monotonic()>=deadline:break
                obs=world.observe()
                if controller:
                    actions=[0]*8
                    for i in np.flatnonzero(world.alive):actions[i]=controller.decide(int(i),obs[i])['action']
                else:actions=(random_policy if method=='random' else greedy_policy)(obs,rng)
                _,reward,_,info=world.step(actions)
                reward_sum+=float(reward.sum());food+=int(info['ate'].sum());steps+=1;used+=1
            rows.append({'method':method,'seed':seed,'joint_steps':steps,'surviving_agent_steps':reward_sum,
                         'food_consumed':food,'alive_at_end':int(world.alive.sum()),'complete':world.done})
            if not world.done:
                return {'comparisons_complete':False,'means':None,'episodes':rows,'joint_steps':used,'stop_reason':'step/runtime limit; partial scores must not be compared'}
    means={method:float(np.mean([r['surviving_agent_steps'] for r in rows if r['method']==method])) for method in ['random','greedy','no_planning','planning']}
    return {'comparisons_complete':True,'means':means,'episodes':rows,'joint_steps':used,'stop_reason':'all configured held-out episodes complete','claim':'No statistical or learning-improvement claim from this engineering evaluation'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['collect','train','evaluate'])
    parser.add_argument('--config',default=str(Path(survival_agent.__file__).parent/'configs/from-scratch.json'),help='Frozen encoder config; scratch default for bounded local checks')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,help='Resume for train; required for evaluate (trusted local file only)')
    parser.add_argument('--max-joint-steps',type=int,default=8)
    parser.add_argument('--max-seconds',type=float,default=10)
    parser.add_argument('--horizon',type=int,default=64)
    parser.add_argument('--rollout-steps',type=int,default=4)
    parser.add_argument('--eval-seeds',default='1001,1002')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new folder')
    if args.max_joint_steps<1 or args.max_seconds<=0:parser.error('Positive step/runtime limits required')
    if args.command=='evaluate' and args.checkpoint is None:parser.error('--checkpoint required for evaluation')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.monotonic();deadline=start+args.max_seconds
    backend=LocalLanguageBackend(read_config(args.config))
    state=checkpoint_io.load(args.checkpoint,map_location='cpu',weights_only=True) if args.checkpoint else None
    args.output.mkdir(parents=True)
    if args.command=='evaluate':
        report=evaluate(backend,state,tuple(int(s) for s in args.eval_seeds.split(',')),args.horizon,args.max_joint_steps,max(.000001,deadline-time.monotonic()))
    else:
        config=TrainConfig(**state['config']) if state else TrainConfig(horizon=args.horizon,rollout_steps=args.rollout_steps)
        trainer=Trainer(backend,config)
        if state:trainer.load_state(state)
        history=[];batches=[];initial=trainer.joint_steps
        while trainer.joint_steps<args.max_joint_steps and time.monotonic()<deadline:
            batch=trainer.collect(args.max_joint_steps-trainer.joint_steps,deadline)
            if batch is None:break
            if args.command=='train':history.append(trainer.update(batch))
            else:batches.append(batch)
        trainer.save(args.output/'checkpoint.safe')
        if batches:checkpoint_io.save({'format':1,'backend':backend.provenance,'config':asdict(config),'source_hashes':source_hashes(),'batches':batches,'use':'Observed on-policy transitions/features; not an offline actor-critic replay trainer'},args.output/'transitions.safe')
        report={'command':args.command,'new_joint_steps':trainer.joint_steps-initial,'cumulative_joint_steps':trainer.joint_steps,
                'optimizer_updates':trainer.updates,'losses':history,'backend':backend.provenance,'train_config':asdict(config),
                'encoder_frozen':True,'learning_claim':False,'stop_reason':'cumulative step cap' if trainer.joint_steps>=args.max_joint_steps else 'cooperative runtime limit'}
    report|={'elapsed_seconds_including_model_load':time.monotonic()-start,'requested_max_seconds':args.max_seconds,
             'backend_provenance':backend.provenance,'source_hashes':source_hashes(),'versions':runtime_versions(),
             'input_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
             'evaluation_horizon':args.horizon if args.command=='evaluate' else None,
             'runtime_limit':'Cooperative, checked before atomic joint steps; a step/update/load can overrun. Use shell timeout for a hard process cap.',
             'cost_usd':0}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
