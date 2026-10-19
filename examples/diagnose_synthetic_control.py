"""Frozen-checkpoint action semantics audit. Does not train or tune any weights."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from survival_agent import checkpoint_io
from survival_agent.hybrid.control import PosteriorActorCritic
from survival_agent.hybrid.world_model import WorldModel
from survival_adapters.synthetic_control import Bridge, SPEC, MODEL, evaluate, SyntheticTrainer
from survival_adapters.training import survival_targets

def main(checkpoint, environment, output):
    checkpoint=Path(checkpoint);output=Path(output)
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True,mode=0o700)
    protocol={'experiment':'Frozen synthetic-control-001 action semantics diagnosis',
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'diagnostic_seeds':list(range(450000,450008)),
        'final_seeds':list(range(660000,660016)),
        'training_updates':0,'checkpoint_selection':'original 001 selection held fixed; no reselection',
        'variants':['trained_greedy','trained_sampled','untrained_greedy','untrained_sampled','random','scripted'],
        'sample_seed':'each world seed + 11, fixed independent torch generator; paired between trained/untrained',
        'selection_or_tuning_from_final':False,'environment_changed':False,'reward_changed':False,'cost_usd':0}
    assert protocol['checkpoint_sha256']=='4fcaa44a9276c372d8d057855a2d7fa59f2a11693c8a7370069d90535ddfa85d'
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();torch.set_num_threads(1)
    saved=checkpoint_io.load(checkpoint)
    world=WorldModel(SPEC,MODEL)
    world.load_state_dict({k.removeprefix('world_model.'):v for k,v in saved['model'].items() if k.startswith('world_model.')})
    # This is exactly the original actor initialization, after world-model fitting.
    torch.manual_seed(123);untrained=PosteriorActorCritic(deepcopy(world))
    trained=PosteriorActorCritic(world);trained.load_state_dict(saved['model'])
    initial_digest={k:v.clone() for k,v in trained.state_dict().items()}
    bridge=Bridge(environment)
    result={}
    try:
        for split in ['diagnostic','final']:
            rows={}
            for variant in protocol['variants']:
                method,mode=variant.split('_') if '_' in variant else (variant,'greedy')
                rows[variant],_=evaluate(bridge,untrained if method=='untrained' else trained,
                    protocol[split+'_seeds'],method,action_mode=mode)
            result[split]=rows
            print(json.dumps({'split':split,'scores':{k:v['mean_per_agent_alive_step_return'] for k,v in rows.items()}}),flush=True)
        # Inspect one real training rollout at the preserved checkpoint, no update.
        trainer=SyntheticTrainer(bridge,trained);batch=trainer.collect()
        logits,values=trained.predict(batch['features'].flatten(0,1))
        active=batch['active'].flatten()
        targets=survival_targets(batch['rewards'],batch['terminated'],batch['truncated'],
            batch['next_values'],batch['active'],gamma=.99).flatten()
        raw_advantage=targets-values.detach();a=raw_advantage[active]
        advantage=(raw_advantage-a.mean())/(a.std(unbiased=False)+1e-6)
        logprob=logits.log_softmax(-1)
        actor=-(logprob.gather(1,batch['actions'].flatten()[:,None]).squeeze(1)[active]*advantage[active]).mean()
        critic=.5*((values-targets)[active]**2).mean()
        actor_grad=torch.autograd.grad(actor,list(trained.actor.parameters()),retain_graph=True)
        critic_grad=torch.autograd.grad(critic,list(trained.value.parameters()))
        norm=lambda gs:float(torch.sqrt(sum((g*g).sum() for g in gs)))
        audit={'real_rollout_joint_steps':16,'active_agent_steps':int(active.sum()),
            'reward_min':float(batch['rewards'].min()),'reward_max':float(batch['rewards'].max()),
            'terminated_count':int(batch['terminated'].sum()),'truncated_count':int(batch['truncated'].sum()),
            'mean_value':float(values[active].detach().mean()),'mean_n_step_target':float(targets[active].mean()),
            'actor_gradient_norm':norm(actor_grad),'half_critic_gradient_norm':norm(critic_grad),
            'optimizer_steps_performed':0,'inactive_targets_zero':bool((targets[~active]==0).all())}
        for k,v in initial_digest.items():torch.testing.assert_close(v,trained.state_dict()[k],rtol=0,atol=0)
        logs=[json.loads(x) for x in (checkpoint.parent/'updates.jsonl').read_text().splitlines()]
        optimization={}
        for key in ['entropy','critic_loss','actor_loss','gradient_norm']:
            a=np.asarray([x[key] for x in logs]);optimization[key]={'first8_mean':float(a[:8].mean()),
                'last8_mean':float(a[-8:].mean()),'minimum':float(a.min()),'maximum':float(a.max()),'all_finite':bool(np.isfinite(a).all())}
        report={'protocol':protocol,'results':result,'target_gradient_audit':audit,'original_optimization':optimization,
            'elapsed_seconds':time.perf_counter()-start,'checkpoint_weights_unchanged':True,
            'browser_inventory':{'apps':[],'browsers':[]},'browser_layout_hydration_verified':False}
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        for p in output.iterdir():p.chmod(0o600)
        print(json.dumps({'audit':audit,'seconds':report['elapsed_seconds']}),flush=True)
    finally:bridge.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--environment',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.checkpoint,args.environment,args.output)
