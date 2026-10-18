"""Regression tests for checkpoint parser boundaries and replay HTML escaping."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import torch
from safetensors.torch import save as tensors_save
import checkpoint_io
from replay import write_replay

class SecurityTests(unittest.TestCase):
    def test_structured_rng_optimizer_arrays_roundtrip(self):
        obj={'tensor':torch.arange(3),'array':np.array([1,2],dtype=np.int64),'tuple':(1,'x'),
             'optimizer':{0:{'step':torch.tensor(1.)}},'rng':np.random.default_rng(0).bit_generator.state}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.safe';checkpoint_io.save(obj,p);out=checkpoint_io.load(p)
        torch.testing.assert_close(obj['tensor'],out['tensor']);np.testing.assert_array_equal(obj['array'],out['array'])
        self.assertEqual(out['tuple'],(1,'x'));self.assertIn(0,out['optimizer']);self.assertEqual(obj['rng'],out['rng'])

    def test_pickle_legacy_and_compressed_archives_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.safe';p.write_bytes(pickle.dumps({'old':'checkpoint'}))
            with self.assertRaises(ValueError):checkpoint_io.load(p)
            with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('metadata.json','{}');z.writestr('tensors.safetensors',tensors_save({}))
            with self.assertRaises(ValueError):checkpoint_io.load(p)

    def test_dynamic_types_and_traversal_members_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.safe'
            root={'format':'survival-safe-checkpoint','version':1,'value':{'type':'execute','module':'os','function':'system'}}
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('metadata.json',json.dumps(root));z.writestr('tensors.safetensors',tensors_save({}))
            with self.assertRaises(ValueError):checkpoint_io.load(p)
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('../outside','payload');z.writestr('metadata.json','{}')
            with self.assertRaises(ValueError):checkpoint_io.load(p)
            self.assertFalse((Path(d).parent/'outside').exists())

    def test_replay_embedded_data_cannot_close_script_tag(self):
        frames=[{'tick':0,'positions':[],'alive':[],'hunger':[],'food':[],
                 'untrusted':'</script><img src=x onerror=alert(1)>'}]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'replay.html';write_replay(p,frames);html=p.read_text()
        self.assertEqual(html.count('</script>'),1);self.assertNotIn('<img src=x',html)
        self.assertIn(r'\u003c/script\u003e',html)

if __name__=='__main__':unittest.main()
