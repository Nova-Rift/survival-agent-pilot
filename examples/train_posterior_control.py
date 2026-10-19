"""Train and evaluate a real CPU policy, then record the predetermined seed."""
import argparse
from pathlib import Path
from survival_adapters.posterior_control import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
