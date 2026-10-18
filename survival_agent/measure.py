"""Bounded CPU feasibility measurements. Never launches remote resources."""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import resource
import time
import numpy as np
import torch
from .simulator import Config, SurvivalWorld, random_policy, greedy_policy
from .agent import HistoryAgent, history_start, history_append, smoke_update, save_checkpoint, load_checkpoint
from .replay import write_replay

def timed(fn, repeats):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
    return {'calls':repeats,'median_ms':float(np.median(samples)*1000),
            'p95_ms':float(np.percentile(samples,95)*1000),'total_seconds':sum(samples)}

def baseline(policy,seeds):
    episodes=[];frames=None;start=time.perf_counter()
    for seed in range(seeds):
        world=SurvivalWorld(seed);rng=np.random.default_rng(1000+seed)
        obs=world.observe();survival=0;food=0;steps=0;replay=[world.frame()]
        while not world.done:
            actions=policy(obs,rng);obs,reward,_,info=world.step(actions)
            survival+=int(reward.sum());food+=int(info['ate'].sum());steps+=1
            if seed==0:replay.append(world.frame())
        episodes.append({'seed':seed,'joint_steps':steps,'surviving_agent_steps':survival,
                         'food_consumed':food,'alive_at_end':int(world.alive.sum()),
                         'horizon_reached':world.tick>=world.config.horizon})
        if seed==0:frames=replay
    elapsed=time.perf_counter()-start
    steps=sum(e['joint_steps'] for e in episodes)
    return {'episodes':episodes,'elapsed_seconds':elapsed,'joint_steps':steps,
            'joint_steps_per_second':steps/elapsed,
            'mean_surviving_agent_steps_per_episode':float(np.mean([e['surviving_agent_steps'] for e in episodes])),
            'interpretation':'Descriptive hand-coded/random baseline only; no learned-agent or continual-learning evidence.'},frames


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='artifacts');args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    start=time.perf_counter()
    report={'kind':'CPU engineering feasibility; no training study',
            'recorded_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
            'torch':torch.__version__,'torch_threads':torch.get_num_threads(),
            'config':vars(Config()),'step_definition':'One joint world transition for up to 8 agents, including local observation generation. Not 8 independent environments.',
            'cost_usd':0,'baselines':{}}
    for name,policy in [('random',random_policy),('local_greedy',greedy_policy)]:
        stats,frames=baseline(policy,12);report['baselines'][name]=stats
        if name=='local_greedy':
            (out/'greedy-replay.json').write_text(json.dumps(frames))
            write_replay(out/'replay.html',frames)
    # Environment + observation cost, excluding baseline-policy selection.
    world=SurvivalWorld(77)
    def env_step():
        nonlocal world
        if world.done:world=SurvivalWorld(77)
        world.step(np.zeros(8,dtype=np.int64))
    report['environment_step_including_resets']=timed(env_step,512)
    model=HistoryAgent().eval();world=SurvivalWorld(5);rng=np.random.default_rng(5)
    history=history_start(world.observe());tensor=torch.from_numpy(history)
    report['model_parameters']=sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for _ in range(3):model.predict(model.represent(tensor));model.plan(tensor)
        report['shared_policy_batch_8']=timed(lambda:model.predict(model.represent(tensor)),40)
        report['depth_two_search_batch_8']=timed(lambda:model.plan(tensor),20)
    report['search_definition']='All 5 root actions, top 2 child actions by prior; 15 latent transitions per agent. Untrained model, no MCTS.'
    def untrained_step():
        nonlocal world,history
        if world.done:world=SurvivalWorld(5);history=history_start(world.observe())
        actions,_=model.plan(torch.from_numpy(history));obs,_,_,_=world.step(actions.numpy())
        history=history_append(history,obs)
    model.eval();report['untrained_search_plus_world_step']=timed(untrained_step,32)
    # Exactly ONE optimizer update to check the computation/checkpoint path.
    world=SurvivalWorld(22);history=history_start(world.observe());actions=random_policy(world.observe(),rng)
    teacher=greedy_policy(world.observe(),rng);next_obs,reward,_,_=world.step(actions)
    next_history=history_append(history,next_obs)
    batch={'history':torch.from_numpy(history),'actions':torch.from_numpy(actions),
           'next_history':torch.from_numpy(next_history),'rewards':torch.from_numpy(reward),
           'alive':torch.from_numpy(world.alive.astype(np.float32)),'teacher_actions':torch.from_numpy(teacher)}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    t=time.perf_counter();losses=smoke_update(model,optimizer,batch)
    report['one_update_smoke_test']={'seconds':time.perf_counter()-t,'losses':losses,
        'interpretation':'One greedy-imitation/latent/reward/value engineering update; not evidence of learning or convergence.'}
    checkpoint=out/'engineering-smoke.safe';save_checkpoint(checkpoint,model,optimizer,world,rng,next_history,1)
    restored,_,rworld,_,rh,updates=load_checkpoint(checkpoint)
    model.eval();restored.eval()
    with torch.no_grad():
        a,q=model.plan(torch.from_numpy(next_history));b,rq=restored.plan(torch.from_numpy(rh))
    assert torch.equal(a,b) and torch.equal(q,rq) and world.frame()==rworld.frame() and updates==1
    report['checkpoint_reload_exact']=True
    report['checkpoint_bytes']=checkpoint.stat().st_size
    report['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report['measurement_body_seconds']=time.perf_counter()-start
    report['limitations']=['12 fixed seeds are descriptive, not a powered study.',
        'No learned survival improvement or cross-world retention measured.',
        'No GPU, parallel-simulator or large-population throughput measured.',
        'No pretrained LLM, language interface or real MuZero MCTS.',
        'One CPU thread; reported per-call p95 is descriptive timing variability, not a research confidence interval.',
        'PyTorch process memory includes runtime/library overhead; timings exclude Python import/startup and remote provisioning.']
    (out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='baselines'},indent=2))
    print('BASELINES',json.dumps({k:{a:b for a,b in v.items() if a!='episodes'} for k,v in report['baselines'].items()}))

if __name__=='__main__':main()
