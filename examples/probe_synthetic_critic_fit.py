"""Fit copied linear value head to fixed existing rollout targets; no actor train."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from survival_agent import checkpoint_io
from survival_agent.hybrid.control import PosteriorActorCritic
from survival_agent.hybrid.world_model import WorldModel
from survival_adapters.synthetic_control import SPEC,MODEL,Bridge,SyntheticTrainer
from survival_adapters.training import survival_targets

def main(checkpoint,environment,output):
    output=Path(output)
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True,mode=0o700)
    start=time.perf_counter();torch.set_num_threads(1)
    agent=PosteriorActorCritic(WorldModel(SPEC,MODEL))
    agent.load_state_dict(checkpoint_io.load(checkpoint)['model'])
    sets=[]
    for seed in [111000,112000]:
        bridge=Bridge(environment);trainer=SyntheticTrainer(bridge,agent);trainer.next_seed=seed
        xs=[];ys=[]
        try:
            for _ in range(4):
                batch=trainer.collect();mask=batch['active'].flatten()
                xs.append(batch['features'].flatten(0,1)[mask])
                ys.append(survival_targets(batch['rewards'],batch['terminated'],batch['truncated'],
                    batch['next_values'],batch['active'],gamma=.99).flatten()[mask])
        finally:bridge.close()
        sets.append((torch.cat(xs),torch.cat(ys)))
    x,y=sets[0];xd,yd=sets[1];head=deepcopy(agent.value)
    optimizer=torch.optim.Adam(head.parameters(),lr=.001);rows=[]
    for i in range(4097):
        if i in [0,128,1024,4096]:
            with torch.no_grad():rows.append({'updates':i,
                'train_mse':float((head(x).squeeze()-y).square().mean()),
                'dev_mse':float((head(xd).squeeze()-yd).square().mean()),
                'mean_value':float(head(xd).mean()),'target_mean':float(yd.mean())})
        if i==4096:break
        loss=.5*(head(x).squeeze()-y).square().mean()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
    xx=torch.cat([x,torch.ones(len(x),1)],1).double()
    xxd=torch.cat([xd,torch.ones(len(xd),1)],1).double()
    solution=torch.linalg.lstsq(xx,y.double(),driver='gelsd').solution
    report={'probe':'fixed existing n-step targets; critic only, original checkpoint unchanged',
        'fit_world_seed_start':111000,'dev_world_seed_start':112000,'fit_samples':len(x),'dev_samples':len(xd),
        'rows':rows,'least_squares_train_mse':float((xx@solution-y).square().mean()),
        'least_squares_dev_mse':float((xxd@solution-yd).square().mean()),
        'feature_rank':int(torch.linalg.matrix_rank(xx)),'seconds':time.perf_counter()-start,'paid_cost_usd':0}
    path=output/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');path.chmod(0o600)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--environment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.checkpoint,a.environment,a.output)
