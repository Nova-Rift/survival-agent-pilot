"""CPU-only, bounded real-control run on the canonical browser environment."""
import argparse
from survival_adapters.synthetic_control import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--environment',required=True)
    args=parser.parse_args()
    run(args.output,args.environment)
