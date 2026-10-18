import checkpoint_io
"""A bounded local MuZero-style episode-replay/unroll smoke command, not a study."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from muzero import MuZeroModel,MuZeroLearner,collect_episode
from language_backend import LocalLanguageBackend,read_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path,help='Trusted local checkpoint; restores model/optimizer/replay/RNG')
    parser.add_argument('--backend-config',help='Explicit local language config; absent means trainable scratch observation encoder')
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--simulations',type=int,default=8)
    parser.add_argument('--updates',type=int,default=1)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-seconds',type=float,default=10)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new run folder')
    if not 1<=args.steps<=16 or not 1<=args.simulations<=16 or not 0<=args.updates<=8 or args.max_seconds<=0:parser.error('Use steps/simulations 1..16, updates 0..8 and positive runtime cap')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.monotonic();deadline=start+args.max_seconds;backend=None
    if args.backend_config:backend=LocalLanguageBackend(read_config(args.backend_config))
    saved=checkpoint_io.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    if saved:
        model=MuZeroModel(**saved['spec'])
        if model.spec['encoder_mode']=='frozen_language_features' and (backend is None or backend.provenance!=model.spec['encoder_provenance']):parser.error('Resume requires the matching explicit frozen local backend')
    elif backend:
        model=MuZeroModel('frozen_language_features',backend.feature_width,args.seed,backend.provenance)
    else:model=MuZeroModel(seed=args.seed)
    learner=MuZeroLearner(model,seed=model.spec['seed'])
    if saved:learner.load_state(saved)
    seed=model.spec['seed']+learner.episodes_collected
    episodes,collection=collect_episode(model,seed,args.steps,args.simulations,deadline,backend)
    if episodes:
        for episode in episodes:learner.replay.add(episode)
        learner.episodes_collected+=1
    losses=[]
    for _ in range(args.updates):
        if not learner.replay.episodes or time.monotonic()>=deadline:break
        losses.append(learner.update())
    args.output.mkdir(parents=True)
    temporary=args.output/'checkpoint.tmp';checkpoint_io.save(learner.state(),temporary);os.replace(temporary,args.output/'checkpoint.safe')
    report={'kind':'Minimal MuZero-style local engineering smoke; not a learning study',
        'model_spec':model.spec,'collection':collection,'requested_simulations_per_decision':args.simulations,
        'new_optimizer_updates':len(losses),'cumulative_optimizer_updates':learner.updates,
        'episode_replay_size':len(learner.replay.episodes),'losses':losses,
        'elapsed_seconds_including_model_load':time.monotonic()-start,
        'runtime_cap':'Cooperative before joint steps/updates; use shell timeout for hard process limit',
        'scratch_encoder_trainable':model.spec['encoder_mode']=='scratch_history',
        'pretrained_encoder_training':'frozen only; fine-tuning explicitly unsupported',
        'scientific_claim':False,'cost_usd':0}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
