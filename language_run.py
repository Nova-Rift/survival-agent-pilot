"""Bounded optional language-wrapper episode, using only existing local models.
This runner never downloads anything and does not invoke latent/MuZero planning.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import torch
from simulator import SurvivalWorld, Config
from communication import LocalCommunication
from language_backend import LocalLanguageBackend, LanguageController, read_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs'/'pretrained.json'))
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.steps<=8:parser.error('Local language-wrapper limit: 1..8 joint steps')
    if args.output.exists():parser.error('Output exists; choose a new folder')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=read_config(args.config);backend=LocalLanguageBackend(config)
    controller=LanguageController(backend,seed=config.seed)
    world=SurvivalWorld(config.seed,replace(Config(),horizon=args.steps));channel=LocalCommunication(world)
    frames=[world.frame()];events=[];food=0;start=time.perf_counter()
    for _ in range(args.steps):
        obs=world.observe();actions=[0]*8;utterances={};decisions=[]
        for i in range(8):
            if not world.alive[i]:continue
            decision=controller.decide(i,obs[i],channel.inbox(i));actions[i]=decision['action']
            if decision['message'] is not None:utterances[i]=decision['message']
            decisions.append({'agent':i,**decision})
        _,rewards,done,info=channel.step(actions,utterances)
        food+=int(info["ate"].sum())
        events.append({'tick':world.tick,'decisions':decisions,'survival_reward':rewards.tolist(),
                       'delivered':{str(i):[asdict(m) for m in channel.inbox(i)] for i in range(8)}})
        frames.append(world.frame())
        if done:break
    summary={'kind':'Local inference action wrapper; not MuZero training or learning evidence',
             'provenance':backend.provenance,'joint_steps':len(events),'config':asdict(world.config),
             'surviving_agent_steps':sum(sum(e['survival_reward']) for e in events),
             'food_consumed':food,'alive_at_end':int(world.alive.sum()),'complete_episode':world.done,
             'invalid_decisions':sum(not d['valid'] for e in events for d in e['decisions']),
             'elapsed_seconds':time.perf_counter()-start,'events':events,'cost_usd':0}
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'frames.json').write_text(json.dumps(frames)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='events'},indent=2))

if __name__=='__main__':main()
