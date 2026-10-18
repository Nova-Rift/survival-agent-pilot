from survival_agent import checkpoint_io
"""Bounded, untrained integrated controller rollout. Default is local pretrained.
Use tiny scratch mode for cheap engineering checks; no training/downloads occur.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import torch
from survival_env.simulator import SurvivalWorld,Config
from survival_env.communication import LocalCommunication
from survival_agent.integrated import IntegratedController
from survival_agent.language_backend import LocalLanguageBackend,read_config

def episode(backend,steps=1,planning=True,speech=False,seed=0):
    if not 1<=steps<=8:raise ValueError('Bounded local milestone: 1..8 joint steps')
    controller=IntegratedController(backend,seed=seed,planning=planning,speech=speech)
    world=SurvivalWorld(seed,replace(Config(),horizon=steps));channel=LocalCommunication(world)
    events=[];frames=[world.frame()];start=time.perf_counter();food=0
    for _ in range(steps):
        observations=world.observe();actions=[0]*8;utterances={};decisions=[]
        for i in range(8):
            if not world.alive[i]:continue
            result=controller.decide(i,observations[i],channel.inbox(i))
            actions[i]=result['action'];decisions.append({'agent':i,**result})
            if result['message'] is not None:utterances[i]=result['message']
        _,reward,done,info=channel.step(actions,utterances);food+=int(info['ate'].sum())
        events.append({'tick':world.tick,'decisions':decisions,'survival_reward':reward.tolist()});frames.append(world.frame())
        if done:break
    summary={'kind':'Untrained integrated controller engineering rollout; no learning evidence',
        'backend':backend.provenance,'planning':planning,'speech':speech,'joint_steps':len(events),
        'config':asdict(world.config),'surviving_agent_steps':sum(sum(e['survival_reward']) for e in events),
        'food_consumed':food,'alive_at_end':int(world.alive.sum()),'elapsed_seconds':time.perf_counter()-start,
        'trainable_head_parameters':sum(p.numel() for p in controller.core.parameters()),
        'optimizer_updates':0,'cost_usd':0,'events':events}
    return summary,frames,{'controller':controller.state(),'channel':channel.state()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).parent.parent/'survival_agent'/'configs/pretrained.json'))
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--no-planning',action='store_true')
    parser.add_argument('--speech',action='store_true',help='Optional slow text generation; cannot override chosen action')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; use a new run folder')
    if not 1<=args.steps<=8:parser.error('Steps must be 1..8')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=read_config(args.config);backend=LocalLanguageBackend(config)
    summary,frames,state=episode(backend,args.steps,not args.no_planning,args.speech,config.seed)
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'frames.json').write_text(json.dumps(frames)+'\n')
    checkpoint_io.save(state,args.output/'engineering-state.safe')
    print(json.dumps({k:v for k,v in summary.items() if k!='events'},indent=2))

if __name__=='__main__':main()
