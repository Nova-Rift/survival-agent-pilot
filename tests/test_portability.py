"""Fresh-process proofs that the two public package boundaries are usable."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class PortabilityTests(unittest.TestCase):
    def run_blocked(self, blocked, body):
        code = '''import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in BLOCKED):
            raise AssertionError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
'''.replace('BLOCKED', repr(blocked)) + body
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_environment_policy_and_runner_without_agent_or_ml(self):
        self.run_blocked(['torch', 'transformers', 'safetensors', 'survival_agent', 'survival_adapters'], '''
import runpy
from survival_env.simulator import Config, SurvivalWorld
from survival_env.run import run
runpy.run_module('examples.custom_policy', run_name='__main__')
summary, frames = run(0, 'greedy', 2)
assert summary['joint_steps'] == 2 and len(frames) == 3
''')

    def test_models_and_toy_adapter_without_survival_environment(self):
        self.run_blocked(['survival_env', 'survival_agent.simulator', 'survival_agent.communication',
                          'survival_adapters.collectors', 'survival_adapters.checkpoints'], '''
import runpy
from survival_agent.agent import HistoryAgent
from survival_agent.muzero import MuZeroModel, mcts
from survival_agent import checkpoint_io
runpy.run_module('examples.toy_env_adapter', run_name='__main__')
assert not any(name.startswith('survival_env') for name in sys.modules)
''')

    def test_legacy_environment_imports_refer_to_canonical_classes(self):
        from survival_agent.simulator import SurvivalWorld as legacy
        from survival_env.simulator import SurvivalWorld
        from survival_agent.communication import LocalCommunication as legacy_channel
        from survival_env.communication import LocalCommunication
        self.assertIs(legacy, SurvivalWorld)
        self.assertIs(legacy_channel, LocalCommunication)

    def test_legacy_muzero_cli_resume_without_optimizer_updates(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'first', Path(folder) / 'resumed'
            for output, resume in [(first, None), (second, first / 'checkpoint.safe')]:
                args = [sys.executable, '-B', '-m', 'survival_agent.muzero_run',
                        '--output', str(output), '--steps', '1', '--simulations', '1',
                        '--updates', '0', '--max-seconds', '8']
                if resume:
                    args += ['--resume', str(resume)]
                result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / 'report.json').read_text())
                self.assertEqual(report['cumulative_optimizer_updates'], 0)
                self.assertTrue((output / 'checkpoint.safe').exists())


if __name__ == '__main__':
    unittest.main()
