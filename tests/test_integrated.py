"""Mock orchestration tests and REAL tiny GPT2/Qwen2 representation tests.
Synthetic saved fixtures have random weights, not pretrained language skills.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import GPT2Config,GPT2LMHeadModel,Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from survival_agent.simulator import SurvivalWorld
from survival_agent.communication import Message
from survival_agent.integrated import IntegratedController
from survival_agent.integrated_run import episode
from survival_agent.language_backend import BackendConfig,LocalLanguageBackend

torch.set_num_threads(1)

@contextmanager
def synthetic_saved_fixture(kind):
    with tempfile.TemporaryDirectory() as d:
        tokenizer=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'food':3,'hello':4},unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]')
        fast.save_pretrained(d)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10)
            if kind=='qwen2':
                model=Qwen2ForCausalLM(Qwen2Config(vocab_size=5,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,intermediate_size=32,max_position_embeddings=512,pad_token_id=1,eos_token_id=2))
            else:
                model=GPT2LMHeadModel(GPT2Config(vocab_size=5,n_positions=512,n_ctx=512,n_embd=16,n_layer=1,n_head=2,pad_token_id=1,eos_token_id=2))
        model.save_pretrained(d,safe_serialization=True)
        yield LocalLanguageBackend(BackendConfig(model_name_or_path=d,max_new_tokens=2))

class FeatureMockBackend:
    feature_width=4
    provenance={'kind':'mock features, no language model'}
    def __init__(self):self.prompts=[]
    def encode(self,prompt):
        self.prompts.append(prompt)
        return torch.tensor([[len(prompt)/1000,prompt.count('F')/10,prompt.count('@'),prompt.count('A')/10]],dtype=torch.float32)
    def generate(self,prompt,seed):return '{"action":4,"message":"try to override"}'

class MockIntegrationTests(unittest.TestCase):
    def test_no_planning_uses_policy_without_search_or_generation(self):
        b=FeatureMockBackend();c=IntegratedController(b,planning=False)
        with (patch.object(c.core,'dynamics',side_effect=AssertionError('Search forbidden')),
              patch.object(b,'generate',side_effect=AssertionError('Implicit text generation forbidden'))):
            result=c.decide(0,SurvivalWorld(0).observe()[0])
        self.assertEqual(result['controller'],'latent_policy_no_planning')
        self.assertIsNone(result['message']);self.assertIn(result['action'],range(5))

    def test_planning_action_comes_from_latent_search_and_text_cannot_override(self):
        c=IntegratedController(FeatureMockBackend(),speech=True)
        with patch.object(c.core,'plan',return_value=(torch.tensor([2]),torch.tensor([[0.,0.,1.,0.,0.]]))) as search:
            result=c.decide(0,SurvivalWorld(0).observe()[0])
        search.assert_called_once();self.assertEqual(result['action'],2)
        self.assertIsNone(result['message']);self.assertFalse(result['speech_valid'])

    def test_agent_state_isolated_reset_and_restore_exact_next_decision(self):
        b=FeatureMockBackend();c=IntegratedController(b,agents=2)
        obs=SurvivalWorld(0).observe()
        c.decide(0,obs[0],[Message(1,0,'PRIVATE-AGENT-0')]);c.decide(1,obs[1])
        self.assertNotIn('PRIVATE-AGENT-0',b.prompts[-1])
        state=c.state();expected=c.decide(0,obs[0])
        restored=IntegratedController(b,agents=2);restored.load_state(state)
        self.assertEqual(expected,restored.decide(0,obs[0]))
        c.reset_agent(0);c.decide(0,obs[0]);self.assertNotIn('PRIVATE-AGENT-0',b.prompts[-1])
        self.assertEqual(c._calls[1],1)
        with self.assertRaises(ValueError):IntegratedController(b,agents=2,planning=False).load_state(state)

class RealModelIntegrationTests(unittest.TestCase):
    def test_real_gpt2_and_qwen2_fixtures_connect_hidden_features_to_world(self):
        for kind in ['gpt2','qwen2']:
            with self.subTest(kind=kind),synthetic_saved_fixture(kind) as backend,patch.object(socket.socket,'connect',side_effect=AssertionError('No network')):
                frozen={k:v.clone() for k,v in backend.model.state_dict().items()}
                for planning in [False,True]:
                    with patch.object(backend,'generate',side_effect=AssertionError('No generation in feature path')):
                        summary,frames,state=episode(backend,steps=2,planning=planning)
                    self.assertEqual(summary['joint_steps'],2);self.assertEqual(len(frames),3)
                    self.assertEqual(summary['optimizer_updates'],0)
                    for event in summary['events']:
                        for d in event['decisions']:
                            self.assertIn(d['action'],range(5));self.assertEqual(len(d['action_scores']),5)
                            self.assertTrue(all(np.isfinite(d['action_scores'])))
                    restored=IntegratedController(backend,planning=planning);restored.load_state(state['controller'])
                for k,v in backend.model.state_dict().items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
                self.assertTrue(all(not p.requires_grad for p in backend.model.parameters()))

    def test_scratch_features_heads_gradient_path_and_deterministic_initialization(self):
        backend=LocalLanguageBackend(BackendConfig(mode='from_scratch',seed=3,max_new_tokens=2))
        c=IntegratedController(backend,seed=3);other=IntegratedController(backend,seed=3)
        for k,v in c.core.state_dict().items():torch.testing.assert_close(v,other.core.state_dict()[k],rtol=0,atol=0)
        features=backend.encode('Only local observations and private history')
        self.assertEqual(features.shape,(1,32));self.assertFalse(features.requires_grad)
        z=c.core.represent(features);policy,value=c.core.predict(z)
        child,reward=c.core.dynamics(z,torch.tensor([1]))
        # Gradient plumbing only: no target, optimizer step or learning claim.
        loss=z.sum()+policy.sum()+value.sum()+child.sum()+reward.sum();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.core.parameters()))
        self.assertTrue(all(p.grad is None for p in backend.model.parameters()))
        a=c.decide(0,SurvivalWorld(0).observe()[0]);b=other.decide(0,SurvivalWorld(0).observe()[0])
        self.assertEqual(a,b)

if __name__=='__main__':unittest.main()
