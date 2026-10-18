"""Survival-specific collection; the model/search package has no world handle."""
from collections import deque
import json
import numpy as np
import torch
from survival_env.simulator import Config, SurvivalWorld
from survival_agent.agent import history_start, history_append
from survival_agent.muzero import Episode, mcts
from survival_agent.integrated import FEATURE_INSTRUCTION
from survival_agent.language_backend import observed_event

def collect_episode(model,seed=0,max_steps=4,simulations=8,deadline=float('inf'),language_backend=None):
    """Bounded self-play collector. Eight equal/shared agents have
    separate local observation histories. Incomplete episodes are NOT replayed.
    Frozen language features must come from an explicit matching local backend.
    """
    language=model.spec['encoder_mode']=='frozen_language_features'
    if language and (language_backend is None or language_backend.provenance!=model.spec['encoder_provenance']):raise ValueError('Matching explicit frozen language encoder required')
    if not 1<=max_steps<=128:raise ValueError('Bounded world horizon required')
    world=SurvivalWorld(seed,Config(horizon=max_steps));obs=world.observe();history=history_start(obs)
    memories=[deque(maxlen=4) for _ in range(8)]
    def current_inputs():
        if not language:return history.copy()
        features=np.zeros((8,model.spec['feature_width']),np.float32)
        for i in np.flatnonzero(world.alive):
            prompt=FEATURE_INSTRUCTION+'\n'+json.dumps({'your_past_turns':list(memories[i]),'current':observed_event(obs[i])},separators=(',',':'))
            features[i]=language_backend.encode(prompt)[0].numpy()
        return features
    inputs=current_inputs()
    rng=np.random.default_rng(seed+10000);records=[{'inputs':[inputs[i].copy()],'actions':[],'rewards':[],
        'policies':[],'root_values':[],'terminated':[],'truncated':[]} for i in range(8)]
    model.eval()
    while not world.done:
        if __import__('time').monotonic()>=deadline:return [],{'complete':False,'joint_steps':world.tick}
        actions=np.zeros(8,np.int64);active=world.alive.copy()
        for i in np.flatnonzero(active):
            search=mcts(model,torch.from_numpy(inputs[i:i+1]),simulations)
            probabilities=search['policy'].astype(np.float64);probabilities/=probabilities.sum()
            actions[i]=rng.choice(5,p=probabilities)
            records[i]['root_values'].append(search['value']);records[i]['policies'].append(search['policy'])
            memories[i].append({'input':observed_event(obs[i]),'your_decision':{'action':int(actions[i]),'message':None}})
        obs,reward,_,info=world.step(actions);history=history_append(history,obs)
        inputs=current_inputs()
        for i in np.flatnonzero(active):
            r=records[i];r['inputs'].append(inputs[i].copy());r['actions'].append(actions[i]);r['rewards'].append(reward[i])
            r['terminated'].append(bool(info['died'][i]));r['truncated'].append(bool(info['truncated'] and world.alive[i]))
    episodes=[]
    with torch.no_grad():
        for i,r in enumerate(records):
            final=0. if r['terminated'][-1] else float(model.initial(torch.from_numpy(r['inputs'][-1][None]))[2][0])
            r['root_values'].append(final)
            ep=Episode(**{k:np.asarray(v,dtype=np.int64 if k=='actions' else bool if k in ['terminated','truncated'] else np.float32) for k,v in r.items()})
            ep.validate();episodes.append(ep)
    return episodes,{'complete':True,'joint_steps':world.tick,'agent_episodes':len(episodes),'seed':seed,
                     'policy_targets':'Actual PUCT child visit frequencies','learning_claim':False}
