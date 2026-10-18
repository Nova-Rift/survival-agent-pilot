"""Inventory and exercise the repository's explicitly allowed documentation commands.

No shell evaluation, dependency installation, downloads or pretrained inference.
All example output paths are redirected to a fresh temporary directory.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SKIP = {
    'export PYTHONPATH="../survival-env:../survival-agent"': 'Source-path prerequisite supplied by the caller; not a dependency installation',
    'python -m venv .venv': 'Isolated setup not exercised; no new environment provisioned',
    '. .venv/bin/activate': 'Activation belongs to the untested isolated setup',
    'python -m pip install -r requirements-language.txt': 'Network/dependency installation excluded; fresh resolution untested',
    'python -B -m scripts.fetch_qwen': 'Manual model download excluded',
    'python -B -m scripts.verify_docs': 'This verifier invocation; do not recurse',
    'python -B -m scripts.verify_docs --no-training': 'This no-training verifier invocation; do not recurse',
}
ALLOWED = {
    'survival_agent.run', 'survival_agent.training', 'survival_agent.muzero_run',
    'survival_agent.language_run', 'survival_agent.integrated_run',
    'examples.custom_policy', 'examples.toy_env_adapter', 'scripts.render_preview',
    'survival_env.run', 'unittest',
}


def inventory():
    items = []
    for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]:
        text = path.read_text()
        for match in re.finditer(r'```bash\n(.*?)```', text, flags=re.S):
            for offset, line in enumerate(match.group(1).splitlines(), 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '\\' in line or any(x in line for x in [';', '&&', '|', '$(', '`']):
                    raise ValueError('Keep documentation commands single-line and directly executable')
                items.append({'file': str(path.relative_to(ROOT)),
                              'line': text[:match.start()].count('\n') + 1 + offset,
                              'command': line})
        # Local Markdown links, including fragments and source line anchors.
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https://', 'http://', '#')):
                continue
            target = target.split('#', 1)[0]
            if not (path.parent / target).exists():
                raise AssertionError(f'Broken local link: {path.name}: {target}')
    return items


def command_args(command, work):
    tokens = shlex.split(command)
    env = os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'PYTHONDONTWRITEBYTECODE': '1'})
    while tokens and '=' in tokens[0]:
        key, value = tokens.pop(0).split('=', 1)
        if key not in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS') or value != '1':
            raise ValueError('Unexpected documented environment setting')
        env[key] = value
    if tokens[:1] == ['timeout']:
        if tokens[1] not in ('20s', '60s'):
            raise ValueError('Unexpected timeout')
        tokens = tokens[2:]
    if tokens[:1] != ['python']:
        raise ValueError('Unknown documentation command: ' + command)
    tokens[0] = sys.executable
    module_index = tokens.index('-m')
    module = tokens[module_index + 1]
    if module not in ALLOWED:
        raise ValueError('Unapproved documentation module: ' + module)
    if module in ('survival_agent.language_run', 'survival_agent.integrated_run') and tokens[-1] != '--help':
        raise ValueError('Only help is allowed for language generation runners')
    for i, token in enumerate(tokens):
        if token.startswith('runs/'):
            tokens[i] = str(work / token)
    return tokens, env


def check_outputs(work, no_training):
    for folder in ['baseline', 'quickstart']:
        path = work / 'runs' / folder
        summary = json.loads((path / 'summary.json').read_text())
        frames = json.loads((path / 'frames.json').read_text())
        assert summary['joint_steps'] == len(frames) - 1
        assert summary['learning_claim'] is False and summary['cost_usd'] == 0
        assert '<canvas' in (path / 'replay.html').read_text()
    if not no_training:
        for folder in ['muzero', 'docs-muzero', 'docs-muzero-resumed',
                       'data', 'ac', 'docs-ac', 'docs-ac-resumed']:
            assert (work / 'runs' / folder / 'checkpoint.safe').is_file(), folder
        for folder in ['muzero', 'docs-muzero', 'docs-muzero-resumed',
                       'data', 'ac', 'eval', 'docs-ac', 'docs-ac-resumed']:
            report = json.loads((work / 'runs' / folder / 'report.json').read_text())
            assert report['cost_usd'] == 0, folder
        report = lambda name: json.loads((work / 'runs' / name / 'report.json').read_text())
        assert report('docs-muzero-resumed')['cumulative_optimizer_updates'] == 2
        assert report('docs-ac')['cumulative_joint_steps'] == 2
        assert report('docs-ac-resumed')['cumulative_joint_steps'] == 3
        assert report('eval')['comparisons_complete'] is True
        assert (work / 'runs' / 'data' / 'transitions.safe').is_file()
    from PIL import Image
    with Image.open(work / 'runs' / 'preview.gif') as image:
        frames = json.loads((work / 'runs' / 'baseline' / 'frames.json').read_text())
        assert image.size == (400, 402) and image.n_frames == len(frames)
        for index in range(image.n_frames):
            image.seek(index)
            image.load()
    provenance = json.loads((work / 'runs' / 'preview-provenance.json').read_text())
    assert provenance['model_inference'] is False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-training',action='store_true',help='Skip training/collector/evaluation experiment commands; regression tests still exercise tiny updates')
    no_training=parser.parse_args().no_training
    items = inventory()
    seen = set()
    with tempfile.TemporaryDirectory(prefix='survival-docs-') as temp:
        work = Path(temp)
        for item in items:
            command = item['command']
            if command in SKIP:
                item.update(status='skipped', reason=SKIP[command])
                continue
            if no_training and any('-m '+name+' ' in command and '--help' not in command for name in ('survival_agent.training','survival_agent.muzero_run')):
                item.update(status='skipped',reason='No-training pass: experiment command excluded; regression suite covers update/resume behavior')
                continue
            if command in seen:
                item.update(status='duplicate', reason='Identical command already passed in this run')
                continue
            args, env = command_args(command, work)
            result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"{item['file']}:{item['line']} {command}\n{result.stdout}\n{result.stderr}")
            if 'examples.custom_policy' in command:
                result_json = json.loads(result.stdout)
                assert result_json['joint_steps'] == 3 and result_json['reward'] == 24
            if 'examples.toy_env_adapter' in command:
                result_json = json.loads(result.stdout)
                assert result_json['terminated'] and result_json['optimizer_updates'] == 0
            if 'unittest' in command:
                summary = re.search(r'Ran (\d+) tests in ([\d.]+)s', result.stderr)
                assert summary and result.stderr.rstrip().endswith('OK')
                item['regression_tests'] = int(summary.group(1))
            seen.add(command)
            item.update(status='passed')
        check_outputs(work, no_training)
        # The simulator and custom scripted policy must work with ML imports blocked.
        code = """import sys, importlib.abc, runpy
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','transformers','safetensors'):
            raise AssertionError('Environment example imported ML dependency: '+fullname)
sys.meta_path.insert(0, BlockML())
runpy.run_module('examples.custom_policy', run_name='__main__')
"""
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True,
                       capture_output=True, text=True, timeout=15)
    print(json.dumps({'commands': items, 'output_assertions': 'passed',
                      'local_markdown_links': 'passed', 'environment_without_ml_imports': 'passed',
                      'temporary_outputs': 'removed', 'fresh_dependency_install': 'not tested', 'no_training_experiments': no_training}, indent=2))


if __name__ == '__main__':
    main()
