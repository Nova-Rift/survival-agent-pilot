"""Combined survival-world/model checkpoints; integration belongs outside models."""
from dataclasses import asdict
from pathlib import Path
import os
import numpy as np
import torch
from survival_agent import checkpoint_io
from survival_agent.agent import HistoryAgent

def save_checkpoint(path, model, optimizer, world, rng, history, updates):
    path=Path(path);temporary=path.with_suffix('.tmp')
    checkpoint_io.save({'format':1,'spec':model.spec,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'world_config':asdict(world.config),'world':world.state(),'rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'history':history,'updates':updates},temporary)
    os.replace(temporary,path)

def load_checkpoint(path):
    from survival_env.simulator import Config, SurvivalWorld
    # Structured parsing and bounded reconstruction; no pickle fallback.
    state=checkpoint_io.load(path,map_location='cpu',weights_only=True)
    if state['format']!=1:raise ValueError('Unsupported checkpoint')
    config=Config(**state['world_config'])
    if state['spec']['obs_width']!=2*config.radius+1:raise ValueError('Checkpoint observation/world mismatch')
    world=SurvivalWorld(0,config)
    expected=(config.agents,state['spec']['history'],5,state['spec']['obs_width'],state['spec']['obs_width'])
    if not isinstance(state['history'],np.ndarray) or state['history'].shape!=expected or state['history'].dtype!=np.float32:raise ValueError('Invalid checkpoint history')
    model=HistoryAgent(**state['spec']);model.load_state_dict(state['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);optimizer.load_state_dict(state['optimizer'])
    world.load_state(state['world'])
    rng=np.random.default_rng();rng.bit_generator.state=state['rng']
    torch.set_rng_state(state['torch_rng'])
    return model,optimizer,world,rng,state['history'],state['updates']
