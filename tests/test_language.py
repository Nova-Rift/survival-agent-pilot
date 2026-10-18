"""Real tiny model tests are labeled separately from mock controller tests.
Fixtures contain locally RANDOM-created weights, not pretrained language ability.
"""
from dataclasses import replace
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from survival_agent.communication import Message
from survival_agent.simulator import SurvivalWorld
from survival_agent.language_backend import BackendConfig, LocalLanguageBackend, LanguageController, ByteVocabulary, parse_decision, read_config

torch.set_num_threads(1)

class RecordingMockBackend:
    def __init__(self):self.calls=[]
    def generate(self,prompt,seed):
        self.calls.append((prompt,seed));return '{"action":0,"message":null}'

class MockControllerTests(unittest.TestCase):
    def test_memory_isolation_reset_and_seed_order(self):
        b=RecordingMockBackend();c=LanguageController(b,agents=2,history=2,seed=7)
        observations=SurvivalWorld(0).observe()
        c.decide(0,observations[0],[Message(1,0,'PRIVATE-TO-AGENT-ZERO')])
        c.decide(1,observations[1]);self.assertNotIn('PRIVATE-TO-AGENT-ZERO',b.calls[-1][0])
        c.decide(0,observations[0]);self.assertIn('PRIVATE-TO-AGENT-ZERO',b.calls[-1][0])
        c.reset_agent(0);c.decide(0,observations[0]);self.assertNotIn('PRIVATE-TO-AGENT-ZERO',b.calls[-1][0])
        self.assertEqual(b.calls[0][1],b.calls[-1][1])
        self.assertEqual(b.calls[1][1],1000010)
        self.assertEqual(c._calls[1],1);self.assertEqual(len(c._memory[1]),1)
        c.reset_all();self.assertTrue(all(not m for m in c._memory))

    def test_no_capability_metadata_global_map_or_other_history_input(self):
        b=RecordingMockBackend();c=LanguageController(b)
        c.decide(0,SurvivalWorld(4).observe()[0],[Message(3,2,'I am strong')])
        payload=json.loads(b.calls[0][0].split('\n',1)[1])
        self.assertEqual(set(payload['current']['view']),{'local_grid','own_hunger_fraction'})
        self.assertEqual(set(payload['current']['received_messages'][0]),{'sender','sent_tick','text'})
        self.assertEqual(payload['your_past_turns'],[])
        self.assertNotIn('all_agents',b.calls[0][0]);self.assertNotIn('positions',b.calls[0][0])

    def test_action_contract_and_invalid_output_are_explicit(self):
        self.assertEqual(parse_decision('{"action":2,"message":"hello"}')['action'],2)
        for text in ['not JSON','{"action":true}','{"action":9}','{"action":0,"capabilities":{}}','{"action":0,"message":""}']:
            result=parse_decision(text);self.assertFalse(result['valid']);self.assertEqual(result['action'],0)

class RealModelFixtureTests(unittest.TestCase):
    def test_fully_scratch_never_reads_pretrained_assets_and_reproduces_seed(self):
        config=BackendConfig(mode='from_scratch',model_name_or_path='/must-not-read',seed=11,max_new_tokens=2)
        with (patch('transformers.AutoConfig.from_pretrained',side_effect=AssertionError('forbidden pretrained config')),
              patch('transformers.AutoTokenizer.from_pretrained',side_effect=AssertionError('forbidden tokenizer')),
              patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=AssertionError('forbidden weights'))):
            a=LocalLanguageBackend(config);b=LocalLanguageBackend(config);other=LocalLanguageBackend(replace(config,seed=12))
        for k,v in a.model.state_dict().items():torch.testing.assert_close(v,b.model.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,other.model.state_dict()[k]) for k,v in a.model.state_dict().items()))
        self.assertIsInstance(a.tokenizer,ByteVocabulary)
        self.assertEqual(a.tokenizer.decode(a.tokenizer.encode('Hi 世界')),'Hi 世界')
        with patch.object(socket.socket,'connect',side_effect=AssertionError('No network')):
            state=torch.get_rng_state().clone()
            self.assertEqual(a.generate('hello',1),b.generate('hello',1))
            torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)

    def test_real_local_safetensors_load_and_generation_without_network(self):
        # This exercises the REAL Transformers load/inference path using a tiny
        # synthetic saved model. It cannot establish Qwen compatibility/quality.
        with tempfile.TemporaryDirectory() as d:
            tokenizer=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'hello':3,'world':4},unk_token='[UNK]'))
            tokenizer.pre_tokenizer=Whitespace()
            fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
            fast.save_pretrained(d)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(5)
                model=GPT2LMHeadModel(GPT2Config(vocab_size=5,n_positions=128,n_ctx=128,n_embd=16,n_layer=1,n_head=2,pad_token_id=1,eos_token_id=2))
            model.save_pretrained(d,safe_serialization=True)
            with patch.object(socket.socket,'connect',side_effect=AssertionError('No network')):
                backend=LocalLanguageBackend(BackendConfig(model_name_or_path=d,max_new_tokens=2))
                self.assertIsInstance(backend.generate('hello world',8),str)
                self.assertEqual(backend.generate('hello world',8),backend.generate('hello world',8))
            for k,v in model.state_dict().items():torch.testing.assert_close(v,backend.model.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(backend.provenance['mode'],'pretrained') # Loader mode, NOT fixture training provenance.

    def test_missing_pretrained_weights_fail_closed_and_config_switch(self):
        with tempfile.TemporaryDirectory() as d,patch.object(socket.socket,'connect',side_effect=AssertionError('No network')):
            with self.assertRaises(RuntimeError):LocalLanguageBackend(BackendConfig(model_name_or_path=d))
        with self.assertRaises(ValueError):LocalLanguageBackend(BackendConfig(model_name_or_path='https://example.invalid/model'))
        self.assertEqual(BackendConfig().mode,'pretrained')
        self.assertEqual(read_config('survival_agent/configs/pretrained.json').mode,'pretrained')
        self.assertEqual(read_config('survival_agent/configs/from-scratch.json').mode,'from_scratch')

if __name__=='__main__':unittest.main()
