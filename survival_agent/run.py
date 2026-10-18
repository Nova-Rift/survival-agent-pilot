"""Run a single bounded baseline simulation without importing PyTorch."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
from .simulator import Config, SurvivalWorld, random_policy, greedy_policy
from .replay import write_replay

def run(seed, policy_name, steps):
    world=SurvivalWorld(seed,replace(Config(),horizon=steps))
    rng=np.random.default_rng(seed+1000)
    policy={'random':random_policy,'greedy':greedy_policy}[policy_name]
    observations=world.observe();frames=[world.frame()];survival=0;food=0
    start=time.perf_counter()
    while not world.done:
        observations,rewards,_,info=world.step(policy(observations,rng))
        survival+=int(rewards.sum());food+=int(info['ate'].sum());frames.append(world.frame())
    summary={'kind':'Untrained baseline simulation only','policy':policy_name,'seed':seed,
             'config':asdict(world.config),'joint_steps':world.tick,
             'surviving_agent_steps':survival,'food_consumed':food,
             'alive_at_end':int(world.alive.sum()),'elapsed_seconds':time.perf_counter()-start,
             'cost_usd':0,'learning_claim':False}
    return summary,frames

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--policy',choices=['random','greedy'],default='greedy')
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--output',type=Path,required=True,help='A new run folder; existing folders are never overwritten')
    args=parser.parse_args()
    if not 1<=args.steps<=128:parser.error('--steps must be 1..128 for the local pilot')
    if args.seed<0:parser.error('--seed must be nonnegative')
    if args.output.exists():parser.error('--output already exists; choose a new folder')
    summary,frames=run(args.seed,args.policy,args.steps)
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'frames.json').write_text(json.dumps(frames)+'\n')
    write_replay(args.output/'replay.html',frames,args.policy,args.seed)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
