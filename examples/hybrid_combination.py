"""Run a bounded CPU integration check; writes a new private output directory."""
import argparse
import json
from survival_adapters.hybrid_combination import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['CartPole-v1','Pendulum-v1'],default='CartPole-v1')
    parser.add_argument('--output',required=True)
    parser.add_argument('--distillation',action='store_true')
    args=parser.parse_args()
    report=run(args.output,args.task,distillation=args.distillation)
    print(json.dumps({k:report[k] for k in ('real_training_transitions','frozen_checkpoint_single_episode_returns','checkpoint_weights_match','elapsed_seconds')}))


if __name__=='__main__':main()
