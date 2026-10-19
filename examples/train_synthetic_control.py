"""Run the current transformer agent's bounded CPU training/reload smoke.

Eight world-model updates and two actor/critic updates; no final evaluation.
Historical GRU run() remains in survival_adapters.synthetic_control.
"""
import argparse
from survival_adapters.transformer_control import run_smoke

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--environment',required=True)
    args=parser.parse_args()
    run_smoke(args.output,args.environment)
