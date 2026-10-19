"""One preregistered CPU comparison of critic optimization budgets, no redesign."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from survival_agent import checkpoint_io
from survival_agent.hybrid.control import PosteriorActorCritic
from survival_agent.hybrid.world_model import WorldModel
from survival_adapters.synthetic_control import Bridge, SPEC, MODEL, SyntheticTrainer, evaluate

def main(checkpoint,environment,output):
    checkpoint=Path(checkpoint);environment=Path(environment).resolve();output=Path(output)
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True,mode=0o700)
    protocol={'experiment':'bounded critic optimization ratio comparison',
        'architecture':'unchanged frozen RSSM posterior + shared linear actor/value',
        'world_model_seed':0,'frozen_world_source':'original synthetic-control-001 checkpoint',
        'source_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'policy_seeds':[123,456,789],'critic_updates_per_actor_update':[1,8],
        'actor_updates_per_run':128,'joint_steps_per_update':16,'joint_steps_per_run':2048,
        'max_active_agent_steps_per_run':16384,'learning_rate':.001,'gamma':.99,
        'target_ema':.99,'target_refresh':'once per actor update after all critic fits',
        'extra_critic_targets':'same frozen n-step targets as the actor update; inactive masked',
        'policy_action_mode':'sampled','reward':'unchanged survival, terminal reward zero',
        'training_world_seed_start':120000,'development_world_seeds':list(range(700000,700008)),
        'reserved_final_world_seeds':list(range(800000,800016)),
        'candidate_actor_updates':[32,64,128],
        'selection':'highest development survival per run; earliest update ties; freeze all before any final',
        'final_role':'report all arms and references, no tuning or reselection',
        'controls':['matched untrained per policy seed','same fixed uniform random','same fixed local scripted'],
        'deployment_gate':'No automatic deployment. Require extra-fit policy beats scripted, random and matched untrained in mean and majority of paired worlds for every policy seed; reproducible checkpoint reload.',
        'representation_changed':False,'recycling':False,'planning':False,'cpu_only':True,'cost_usd':0}
    assert protocol['source_checkpoint_sha256']=='4fcaa44a9276c372d8d057855a2d7fa59f2a11693c8a7370069d90535ddfa85d'
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=environment,capture_output=True,text=True,check=True).stdout.strip()
    assert commit=='e4d3f66a3f870fefe7340e35f206b587eb0c69e4'
    assert not subprocess.run(['git','status','--porcelain'],cwd=environment,capture_output=True,text=True,check=True).stdout
    protocol['environment_commit']=commit
    protocol['environment_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in environment.glob('*.mjs')}
    import survival_adapters.synthetic_control as implementation
    protocol['adapter_sha256']=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    protocol['experiment_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();torch.set_num_threads(1)
    saved=checkpoint_io.load(checkpoint);world=WorldModel(SPEC,MODEL)
    world.load_state_dict({k.removeprefix('world_model.'):v for k,v in saved['model'].items() if k.startswith('world_model.')})
    frozen_world={k:v.clone() for k,v in world.state_dict().items()}
    evaluator=Bridge(environment);runs=[];untrained={}
    try:
        for seed in protocol['policy_seeds']:
            torch.manual_seed(seed);initial=PosteriorActorCritic(deepcopy(world));untrained[seed]=deepcopy(initial)
            for ratio in protocol['critic_updates_per_actor_update']:
                run_start=time.perf_counter();agent=deepcopy(initial);bridge=Bridge(environment)
                trainer=SyntheticTrainer(bridge,agent,critic_updates=ratio)
                trainer.rng=torch.Generator().manual_seed(seed);trainer.next_seed=120000
                candidates=[];logs=[]
                try:
                    for _ in range(128):
                        logs.append(trainer.update(trainer.collect()))
                        if trainer.updates in protocol['candidate_actor_updates']:
                            dev,_=evaluate(evaluator,agent,protocol['development_world_seeds'],action_mode='sampled')
                            candidates.append({'updates':trainer.updates,'development':dev,'weights':deepcopy(agent.state_dict())})
                    best=max(candidates,key=lambda x:(x['development']['mean_per_agent_alive_step_return'],-x['updates']))
                    agent.load_state_dict(best['weights'])
                    for k,v in frozen_world.items():torch.testing.assert_close(v,agent.world_model.state_dict()[k],rtol=0,atol=0)
                    name=f'seed-{seed}-critic-{ratio}';path=output/(name+'.safe')
                    checkpoint_io.save({'format':'synthetic-critic-comparison-v1','model':agent.state_dict(),
                        'protocol':protocol,'policy_seed':seed,'critic_updates':ratio,'selected_actor_updates':best['updates']},path)
                    runs.append({'name':name,'policy_seed':seed,'critic_updates':ratio,'selected_actor_updates':best['updates'],
                        'selected_development':best['development'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'total_actor_updates':128,'total_critic_updates':128*ratio,'joint_steps':trainer.joint_steps,
                        'active_agent_steps':trainer.active_steps,'training_and_development_seconds':time.perf_counter()-run_start,
                        'development_candidates':[{k:v for k,v in c.items() if k!='weights'} for c in candidates],
                        'updates':logs})
                    print(json.dumps({'phase':'run_frozen','name':name,'selected_update':best['updates'],
                        'development_return':best['development']['mean_per_agent_alive_step_return']}),flush=True)
                finally:bridge.close()
        # All six selections and their hashes persisted before final-world access.
        (output/'selection-frozen.json').write_text(json.dumps({'protocol':protocol,
            'runs':[{k:r[k] for k in ['name','policy_seed','critic_updates','selected_actor_updates','checkpoint_sha256']} for r in runs]},indent=2)+'\n')
        final_seeds=protocol['reserved_final_world_seeds'];controls={}
        for seed,agent in untrained.items():
            controls[f'untrained-{seed}'],_=evaluate(evaluator,agent,final_seeds,'untrained',action_mode='sampled')
        for method in ['random','scripted']:
            controls[method],_=evaluate(evaluator,initial,final_seeds,method)
        for r in runs:
            agent=PosteriorActorCritic(deepcopy(world))
            agent.load_state_dict(checkpoint_io.load(output/(r['name']+'.safe'))['model'])
            final,_=evaluate(evaluator,agent,final_seeds,action_mode='sampled');r['final']=final
            recorded,frames=evaluate(evaluator,agent,[70000],record=True,action_mode='sampled')
            restored=PosteriorActorCritic(deepcopy(world));restored.load_state_dict(checkpoint_io.load(output/(r['name']+'.safe'))['model'])
            repeated,again=evaluate(evaluator,restored,[70000],record=True,action_mode='sampled')
            assert recorded==repeated and frames==again
            r['reload_rollout_match']=True
            (output/(r['name']+'-recording.json')).write_text(json.dumps({'seed':70000,
                'checkpoint_sha256':r['checkpoint_sha256'],'policy_action_mode':'sampled','frames':frames[0]},separators=(',',':'))+'\n')
            own=np.array([x['mean_return'] for x in final['episodes']]);paired={}
            for name in [f"untrained-{r['policy_seed']}",'random','scripted']:
                reference=np.array([x['mean_return'] for x in controls[name]['episodes']]);diff=own-reference
                paired[name]={'mean_difference':float(diff.mean()),'wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'losses':int((diff<0).sum())}
            r['paired_comparisons']=paired
            r['passes_predeclared_gate']=all(x['mean_difference']>0 and x['wins']>8 for x in paired.values())
            print(json.dumps({'phase':'final','name':r['name'],'return':final['mean_per_agent_alive_step_return'],
                'gate':r['passes_predeclared_gate']}),flush=True)
        eligible=all(r['passes_predeclared_gate'] for r in runs if r['critic_updates']==8)
        report={'protocol':protocol,'runs':runs,'controls':controls,'deployment_earned':eligible,
            'elapsed_seconds':time.perf_counter()-start,'cost_usd':0,
            'limitations':['Three policy seeds share one frozen world-model seed','Small engineering budget; no continual-learning evidence',
                'Final worlds used once after all six selections; world is paired comparison unit, not each agent',
                'Critic fitting changes policy trajectories; matched joint-step budgets, active-agent exposure reported separately'],
            'browser_layout_hydration_verified':False}
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for p in output.iterdir():p.chmod(0o600)
        print(json.dumps({'phase':'complete','seconds':report['elapsed_seconds'],'deployment_earned':eligible,
            'control_returns':{k:v['mean_per_agent_alive_step_return'] for k,v in controls.items()}}),flush=True)
    finally:evaluator.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--environment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.checkpoint,a.environment,a.output)
