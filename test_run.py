import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class RunnerTests(unittest.TestCase):
    def test_bounded_run_writes_trace_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'episode'
            command=[sys.executable,'-B','run.py','--seed','1','--steps','8','--output',str(out)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((out/'summary.json').read_text())
            frames=json.loads((out/'frames.json').read_text())
            self.assertEqual(summary['joint_steps'],8);self.assertEqual(len(frames),9)
            self.assertFalse(summary['learning_claim']);self.assertIn('greedy baseline, seed 1',(out/'replay.html').read_text())
            again=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(again.returncode,0);self.assertIn('already exists',again.stderr)

    def test_rejects_unbounded_local_run(self):
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([sys.executable,'-B','run.py','--steps','1000000','--output',str(Path(d)/'bad')],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertFalse((Path(d)/'bad').exists())

if __name__=='__main__':unittest.main()
