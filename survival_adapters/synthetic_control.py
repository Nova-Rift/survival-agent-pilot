"""Bounded CPU posterior actor/critic on the canonical browser synthetic v2.

Uses the existing RSSM, linear actor/value attachment, safe checkpoints and
boundary-aware targets. No learned-prior planning, recycling or legacy rule edits.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from survival_agent import checkpoint_io
from survival_agent.hybrid.contracts import EnvSpec, ActionSpec
from survival_agent.hybrid.replay import SequenceReplay
from survival_agent.hybrid.world_model import WorldModel, ModelConfig, WorldModelTrainer
from survival_agent.hybrid.control import PosteriorActorCritic
from .coverage_data import fixed_target_batch, replay_digest
from .training import survival_targets

SPEC = EnvSpec(200, ActionSpec('discrete', count=5))
MODEL = ModelConfig(deter=32, hidden=32, stoch=4, classes=4)
PROTOCOL = {
    'environment_version':'synthetic-survival-v2.0', 'environment_agents':8,
    'observations':'8 x 5 x 5 local channels, no overhead state or agent ID',
    'model':'frozen categorical RSSM posterior + shared linear actor/critic',
    'world_model_training_worlds':16, 'world_model_updates':300,
    'model_seed':0, 'policy_seed':123, 'policy_updates':128,
    'joint_steps_per_update':16, 'candidate_updates':[32,64,128],
    'validation_seeds':list(range(220000,220008)),
    'final_seeds':list(range(330000,330016)), 'recording_seed':70000,
    'training_seed_start':110000, 'training_seed_end_exclusive':200000,
    'prefit_seed_start':30000, 'selection':'highest validation mean per-agent alive-step return; earliest tie',
    'final_role':'report only after checkpoint selection is frozen',
    'evaluation_action_mode':'sampled',
    'evaluation_action_seed':'world seed + 11; independent fixed torch generator',
    'gamma':.99, 'learning_rate':.001, 'entropy_weight':.01,
    'reward':'1 per agent remaining alive after each transition; zero on death',
    'return_ceiling':95, 'wear_terminal_step':96,
    'planning':False, 'recycling':False, 'cpu_only':True, 'cost_usd':0,
}

class Bridge:
    def __init__(self, environment):
        self.environment = Path(environment).resolve()
        script = Path(__file__).resolve().parents[1]/'scripts/synthetic-bridge.mjs'
        self.process = subprocess.Popen(['node',str(script),str(self.environment)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)

    def call(self, **request):
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line: raise RuntimeError('Environment bridge exited')
        result = json.loads(line)
        if 'error' in result: raise RuntimeError(result['error'])
        if result['version'] != PROTOCOL['environment_version']: raise ValueError('Environment version mismatch')
        obs = np.asarray(result['observations'],dtype=np.float32)
        if obs.shape != (8,200) or not np.isfinite(obs).all(): raise ValueError('Invalid local observations')
        return result

    def reset(self, seed): return self.call(op='reset',seed=int(seed))
    def step(self, actions=None, scripted=False):
        return self.call(op='step',actions=None if actions is None else list(map(int,actions)),scripted=scripted)
    def close(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.terminate(); self.process.wait(timeout=5)
        finally:self.process.stdout.close()

@torch.no_grad()
def observe(model, result, state=None, actions=None):
    first = state is None
    return model.observe_step(model.initial(8) if first else state,
        torch.zeros(8,dtype=torch.long) if actions is None else actions,
        torch.tensor(result['observations'],dtype=torch.float32),
        torch.full((8,),first,dtype=torch.bool),stochastic=False)[0]

def collect_prefit(bridge):
    replay = SequenceReplay(SPEC,capacity=16384,max_bytes=16*1024*1024)
    count = 0
    for episode in range(PROTOCOL['world_model_training_worlds']):
        seed = PROTOCOL['prefit_seed_start']+episode
        rng = np.random.default_rng(seed)
        result = bridge.reset(seed)
        streams = [SequenceReplay(SPEC,capacity=128) for _ in range(8)]
        for i,stream in enumerate(streams):stream.start(result['observations'][i])
        while not result['done']:
            previous = result
            result = bridge.step(rng.integers(5,size=8),scripted=episode%2==1)
            for i,stream in enumerate(streams):
                if not previous['frame']['alive'][i]:continue
                stream.append(result['actions'][i],result['rewards'][i],result['observations'][i],
                    bool(result['info']['died'][i]),bool(result['info']['truncated']))
                count += 1
        for stream in streams:
            assert stream.pending is None and len(stream.episodes)==1
            replay.episodes.extend(stream.episodes)
    assert sum(replay.nbytes(e) for e in replay.episodes)<=replay.max_bytes
    assert count <= replay.capacity
    return replay,count

def policy_actions(logits, *, action_mode, generator=None):
    """Declare deployment semantics; categorical training is not argmax control."""
    if action_mode=='greedy':return logits.argmax(-1)
    if action_mode!='sampled' or generator is None:
        raise ValueError('Sampled policy requires an explicit RNG; mode must be declared')
    return torch.multinomial(logits.softmax(-1),1,generator=generator).squeeze(-1)

@torch.no_grad()
def evaluate(bridge, agent, seeds, method='trained', record=False, *, action_mode='greedy'):
    # Greedy default preserves the original public diagnostic API; current run()
    # explicitly evaluates the stochastic policy actually optimized by collect().
    if method not in ['trained','untrained','random','scripted'] or action_mode not in ['greedy','sampled']:
        raise ValueError('Explicit known policy and action mode required')
    rows, recordings = [],[]
    for seed in seeds:
        result = bridge.reset(seed); state = observe(agent.world_model,result)
        rng=np.random.default_rng(seed+11); returns=np.zeros(8); resources=np.zeros(8,dtype=int)
        action_rng=torch.Generator().manual_seed(seed+11)
        action_counts=np.zeros(5,dtype=int); entropy_sum=0.; max_probability_sum=0.;active_steps=0
        move_attempts=0;stationary_moves=0;feature_sum=np.zeros(48);feature_square_sum=np.zeros(48)
        posterior_modes=set()
        frames=[result['frame']] if record else []
        while not result['done']:
            previous=result;active=np.asarray(previous['frame']['alive'],dtype=bool)
            if method=='random':actions=rng.integers(5,size=8)
            elif method=='scripted':actions=None
            else:
                features=agent.world_model.features(state)
                logits,_=agent.predict(features);probabilities=logits.softmax(-1)
                entropy_sum+=float((-(probabilities*probabilities.log()).sum(-1))[active].sum())
                max_probability_sum+=float(probabilities.max(-1).values[active].sum())
                selected=features.numpy()[active];feature_sum+=selected.sum(0);feature_square_sum+=(selected**2).sum(0)
                for mode in state.stoch.argmax(-1).numpy()[active]:posterior_modes.add(tuple(map(int,mode)))
                actions=policy_actions(logits,action_mode=action_mode,generator=action_rng).numpy()
            result=bridge.step(actions,scripted=method=='scripted')
            chosen=np.asarray(result['actions']);active_steps+=int(active.sum())
            action_counts+=np.bincount(chosen[active],minlength=5)
            moves=active&(chosen!=0);move_attempts+=int(moves.sum())
            same=np.all(np.asarray(previous['frame']['positions'])==np.asarray(result['frame']['positions']),axis=1)
            stationary_moves+=int((moves&same).sum())
            returns+=np.asarray(result['rewards']);resources+=np.asarray(result['info']['ate'],dtype=int)
            state=observe(agent.world_model,result,state,torch.tensor(result['actions'],dtype=torch.long))
            if record:frames.append(result['frame'])
        rows.append({'seed':seed,'per_agent_returns':returns.tolist(),'mean_return':float(returns.mean()),
            'joint_steps':result['frame']['tick'],'failure_causes':result['frame']['failureCauses'],
            'resources_consumed':resources.tolist(),'active_agent_steps':active_steps,
            'action_counts':action_counts.tolist(),'stationary_move_attempts':stationary_moves,'move_attempts':move_attempts,
            'mean_policy_entropy':None if method in ['random','scripted'] else entropy_sum/active_steps,
            'mean_max_action_probability':None if method in ['random','scripted'] else max_probability_sum/active_steps,
            'posterior_mode_combinations':None if method in ['random','scripted'] else len(posterior_modes),
            'mean_feature_variance':None if method in ['random','scripted'] else float(np.maximum(0,feature_square_sum/active_steps-(feature_sum/active_steps)**2).mean())})
        if record:recordings.append(frames)
    mean=float(np.mean([x['mean_return'] for x in rows]))
    return {'method':method,'action_mode':action_mode if method in ['trained','untrained'] else method,
        'seed_count':len(rows),'mean_per_agent_alive_step_return':mean,
        'fraction_of_return_ceiling':mean/95,'episodes':rows},recordings

class SyntheticTrainer:
    def __init__(self,bridge,agent):
        self.bridge,self.agent=bridge,agent
        self.optimizer=torch.optim.Adam(agent.trainable_parameters(),lr=.001)
        self.rng=torch.Generator().manual_seed(123)
        self.next_seed=110000;self.joint_steps=0;self.active_steps=0;self.updates=0
        self.result=None;self.state=None

    @torch.no_grad()
    def collect(self):
        data={k:[] for k in ['features','actions','rewards','terminated','truncated','next_values','active']}
        for _ in range(16):
            if self.result is None or self.result['done']:
                assert self.next_seed<200000
                self.result=self.bridge.reset(self.next_seed);self.next_seed+=1
                self.state=observe(self.agent.world_model,self.result)
            features=self.agent.world_model.features(self.state).detach().clone()
            active=torch.tensor(self.result['frame']['alive'],dtype=torch.bool)
            logits,_=self.agent.predict(features)
            actions=policy_actions(logits,action_mode='sampled',generator=self.rng)
            result=self.bridge.step(actions.numpy())
            next_state=observe(self.agent.world_model,result,self.state,actions)
            # Same-episode true final value, before any reset; inactive slots masked.
            next_values=self.agent.target_value(self.agent.world_model.features(next_state)).squeeze(-1)
            row={'features':features,'active':active,'actions':actions,
                'rewards':torch.tensor(result['rewards'],dtype=torch.float32),
                'terminated':torch.tensor(result['info']['died'],dtype=torch.bool),
                'truncated':torch.full((8,),result['info']['truncated'],dtype=torch.bool),
                'next_values':next_values.clone()}
            for k,v in row.items():data[k].append(v)
            self.result,self.state=result,next_state
            self.joint_steps+=1;self.active_steps+=int(active.sum())
        return {k:torch.stack(v) for k,v in data.items()}

    def update(self,batch):
        # Existing posterior-control learner with explicit inactive-agent masking.
        logits,value=self.agent.predict(batch['features'].flatten(0,1))
        active=batch['active'].flatten(); mask=active.float();denom=mask.sum().clamp_min(1)
        with torch.no_grad():
            returns=survival_targets(batch['rewards'],batch['terminated'],batch['truncated'],
                batch['next_values'],batch['active'],gamma=.99).flatten()
        advantage=returns-value.detach()
        selected=advantage[active]
        advantage=(advantage-selected.mean())/(selected.std(unbiased=False)+1e-6)
        logprob=logits.log_softmax(-1)
        actor=-(logprob.gather(1,batch['actions'].flatten()[:,None]).squeeze(1)*advantage*mask).sum()/denom
        critic=((value-returns).square()*mask).sum()/denom
        entropy=(-(logprob.exp()*logprob).sum(-1)*mask).sum()/denom
        loss=actor+.5*critic-.01*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite synthetic control loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.agent.trainable_parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step();self.agent.update_target(.99);self.updates+=1
        return {'updates':self.updates,'joint_steps':self.joint_steps,'active_agent_steps':self.active_steps,
            'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach()),
            'entropy':float(entropy.detach()),'gradient_norm':float(norm)}

def run(output,environment):
    output=Path(output)
    if output.exists():raise ValueError('Use a fresh experiment output')
    output.mkdir(parents=True,mode=0o700)
    environment=Path(environment).resolve()
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in environment.glob('*.mjs')}
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=environment,capture_output=True,text=True,check=True).stdout.strip()
    assert commit=='e4d3f66a3f870fefe7340e35f206b587eb0c69e4', 'Reconcile changed environment before training'
    assert not subprocess.run(['git','status','--porcelain'],cwd=environment,capture_output=True,text=True,check=True).stdout
    import inspect
    implementation={Path(inspect.getfile(obj)).name:hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
        for obj in [WorldModel,PosteriorActorCritic,survival_targets]}
    implementation['synthetic-bridge.mjs']=hashlib.sha256((Path(__file__).resolve().parents[1]/'scripts/synthetic-bridge.mjs').read_bytes()).hexdigest()
    protocol={**PROTOCOL,'environment_source_sha256':source,'model_config':asdict(MODEL),
        'environment_commit':commit,'implementation_sha256':implementation,
        'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(1);torch.manual_seed(0);start=time.perf_counter()
    bridge=Bridge(environment)
    try:
        replay,transitions=collect_prefit(bridge)
        world=WorldModel(SPEC,MODEL);learner=WorldModelTrainer(world,learning_rate=.001)
        sampler=np.random.default_rng(60000)
        for i in range(300):
            learner.step(fixed_target_batch(replay,sampler))
            if (i+1)%100==0:print(json.dumps({'phase':'world_prefit','updates':i+1}),flush=True)
        torch.manual_seed(123);agent=PosteriorActorCritic(world);untrained=deepcopy(agent)
        frozen_world={k:v.clone() for k,v in world.state_dict().items()}
        trainer=SyntheticTrainer(bridge,agent);candidates=[];logs=[]
        # Evaluation uses a distinct bridge so it cannot mutate the training world.
        evaluator=Bridge(environment)
        try:
            for _ in range(PROTOCOL['policy_updates']):
                logs.append(trainer.update(trainer.collect()))
                if trainer.updates in PROTOCOL['candidate_updates']:
                    validation,_=evaluate(evaluator,agent,PROTOCOL['validation_seeds'],action_mode=PROTOCOL['evaluation_action_mode'])
                    candidates.append({'updates':trainer.updates,'joint_steps':trainer.joint_steps,
                        'active_agent_steps':trainer.active_steps,'validation':validation,'weights':deepcopy(agent.state_dict())})
                    print(json.dumps({'phase':'validation','updates':trainer.updates,'return':validation['mean_per_agent_alive_step_return']}),flush=True)
            best=max(candidates,key=lambda c:(c['validation']['mean_per_agent_alive_step_return'],-c['updates']))
            agent.load_state_dict(best['weights'])
            for k,v in frozen_world.items():torch.testing.assert_close(v,world.state_dict()[k],rtol=0,atol=0)
            checkpoint_io.save({'format':'synthetic-posterior-control-v1','protocol':protocol,
                'model':agent.state_dict(),'selected_updates':best['updates']},output/'selected-policy.safe')
            checkpoint_hash=hashlib.sha256((output/'selected-policy.safe').read_bytes()).hexdigest()
            frozen={'checkpoint_sha256':checkpoint_hash,'selected_updates':best['updates'],
                'selected_validation':best['validation'],'final_seeds':PROTOCOL['final_seeds'],
                'final_role':'report_only_no_selection','recording_seed':70000}
            (output/'selection-frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
            comparisons=[]
            for method in ['trained','untrained','random','scripted']:
                result,_=evaluate(evaluator,untrained if method=='untrained' else agent,PROTOCOL['final_seeds'],method,action_mode=PROTOCOL['evaluation_action_mode'])
                comparisons.append(result)
            recorded,frames=evaluate(evaluator,agent,[70000],record=True,action_mode=PROTOCOL['evaluation_action_mode'])
            restored=PosteriorActorCritic(WorldModel(SPEC,MODEL))
            restored.load_state_dict(checkpoint_io.load(output/'selected-policy.safe')['model'])
            repeated,repeated_frames=evaluate(evaluator,restored,[70000],record=True,action_mode=PROTOCOL['evaluation_action_mode'])
            assert frames==repeated_frames and recorded==repeated
            report={'protocol':protocol,'world_model_real_agent_transitions':transitions,
                'world_model_replay_sha256':replay_digest(replay),'world_model_frozen_during_policy_training':True,
                'total_policy_joint_steps':trainer.joint_steps,'total_policy_active_agent_steps':trainer.active_steps,
                'total_policy_updates':trainer.updates,'selected_updates':best['updates'],
                'validation_candidates':[{k:v for k,v in c.items() if k!='weights'} for c in candidates],
                'checkpoint_sha256':checkpoint_hash,'final_comparisons':comparisons,
                'representative_recording':recorded,'checkpoint_reload_recording_match':True,
                'elapsed_seconds':time.perf_counter()-start,'cost_usd':0,
                'limitations':['One training seed, bounded engineering run; no retention or generalization proof',
                    'Frozen small posterior and linear heads; incomplete Draft-2 model',
                    'Actual synthetic v2 browser rules; no learned-prior planning or recycling',
                    'Performance claims must compare all held-out references; trained does not imply improved']}
            (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            (output/'recording.json').write_text(json.dumps({'environment_version':PROTOCOL['environment_version'],
                'checkpoint_sha256':checkpoint_hash,'seed':70000,'policy':'trained frozen-posterior actor/critic',
                'config':evaluator.reset(70000)['config'],'frames':frames[0]},separators=(',',':'))+'\n')
            (output/'updates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in logs))
            for f in output.iterdir():f.chmod(0o600)
            print(json.dumps({'phase':'complete','selected_updates':best['updates'],'final_scores':
                {x['method']:x['mean_per_agent_alive_step_return'] for x in comparisons},
                'elapsed_seconds':report['elapsed_seconds']}),flush=True)
            return report
        finally:evaluator.close()
    finally:bridge.close()
