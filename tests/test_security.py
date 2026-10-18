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
from survival_agent import checkpoint_io
from survival_agent.replay import write_replay

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

    def test_episode_shapes_dtypes_and_length_rejected_before_broadcast(self):
        from copy import deepcopy
        from tests.test_muzero import one_step_episode
        base=one_step_episode()
        cases=[('terminated',np.ones((1,1),dtype=bool)),
               ('truncated',np.zeros((1,1),dtype=bool)),
               ('root_values',np.zeros((2,1),dtype=np.float32)),
               ('actions',np.zeros(4097,dtype=np.int64)),
               ('actions',np.zeros(1,dtype=np.float32)),
               ('rewards',np.zeros(1,dtype=object))]
        for name,value in cases:
            with self.subTest(field=name,shape=value.shape):
                episode=deepcopy(base);setattr(episode,name,value)
                with self.assertRaises(ValueError):episode.validate()

    def test_model_and_world_dimensions_rejected_before_allocation(self):
        from unittest.mock import patch
        from survival_agent.agent import HistoryAgent
        from survival_agent.integrated import LanguageLatentCore
        from survival_agent.simulator import Config,SurvivalWorld
        with patch('torch.nn.Linear',side_effect=AssertionError('must reject before allocation')):
            for kwargs in [{'width':10**9},{'history':10**9},{'obs_width':10**9}]:
                with self.assertRaises(ValueError):HistoryAgent(**kwargs)
            with self.assertRaises(ValueError):LanguageLatentCore(10**9)
        with patch('numpy.random.default_rng',side_effect=AssertionError('must reject before allocation')):
            with self.assertRaises(ValueError):SurvivalWorld(config=Config(size=10**9))

    def test_resume_rejects_dimensions_and_backend_before_constructor(self):
        from unittest.mock import patch
        from survival_agent.muzero_run import model_from_checkpoint_spec
        spec={'encoder_mode':'frozen_language_features','feature_width':10**9,'seed':0,
              'encoder_provenance':{'test':True},'language_encoder_training':'frozen'}
        with patch('survival_agent.muzero_run.MuZeroModel',side_effect=AssertionError('must reject before construction')):
            with self.assertRaises(ValueError):model_from_checkpoint_spec(spec,None)
            spec['feature_width']=32
            with self.assertRaises(ValueError):model_from_checkpoint_spec(spec,None)
            spec['encoder_mode']='scratch_history'
            with self.assertRaises(ValueError):model_from_checkpoint_spec(spec,object())

    def test_json_overflow_float_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'overflow.safe'
            metadata='{"format":"survival-safe-checkpoint","version":1,"value":1e309}'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('metadata.json',metadata);z.writestr('tensors.safetensors',tensors_save({}))
            with self.assertRaises(ValueError):checkpoint_io.load(p)

    def test_replay_embedded_data_cannot_close_script_tag(self):
        frames=[{'tick':0,'positions':[],'alive':[],'hunger':[],'food':[],
                 'untrusted':'</script><img src=x onerror=alert(1)>'}]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'replay.html';write_replay(p,frames);html=p.read_text()
        self.assertEqual(html.count('</script>'),1);self.assertNotIn('<img src=x',html)
        self.assertIn(r'\u003c/script\u003e',html)

if __name__=='__main__':unittest.main()
