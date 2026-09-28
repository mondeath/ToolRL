"""Optional adapter checks use PyTorch but no model downloads."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.policies import AgentState
from src.policies.huggingface import HuggingFacePolicy, ContextLimitError
from src.policies.schema import ACTION_SCHEMA


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch dependency not installed')
class HuggingFaceAdapterTests(unittest.TestCase):
    def make_policy(self):
        import torch
        class Inputs(dict):
            def to(self, device): return self
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 1
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return 'formatted prompt'
            def __call__(self, prompt, **kwargs):
                return Inputs(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))
            def decode(self, tokens, **kwargs): return str(tokens)
        class Model:
            config = types.SimpleNamespace(max_position_embeddings=128)
            def generate(self, **kwargs):
                return torch.cat([kwargs['input_ids'], torch.randint(2, 100000, (1, 4))], dim=1)
        p = HuggingFacePolicy.__new__(HuggingFacePolicy)
        p.config = {'max_context_tokens':128, 'max_new_tokens':4, 'do_sample':True, 'temperature':0.8, 'top_p':0.95}
        p.tokenizer, p.model, p.device = Tokenizer(), Model(), torch.device('cpu')
        p.seed, p.call_index = 0, 0
        return p

    def test_rng_isolation_and_completion_only(self):
        import torch
        policy = self.make_policy()
        state = AgentState('Test question', [], ACTION_SCHEMA)
        before = torch.random.get_rng_state().clone()
        first = policy.new_session(11)
        a, b = first.act(state), first.act(state)
        self.assertEqual(len(a.token_ids), 4)
        self.assertNotEqual(a.token_ids, b.token_ids)
        self.assertEqual(a.token_ids, policy.new_session(11).act(state).token_ids)
        self.assertNotEqual(a.token_ids, policy.new_session(12).act(state).token_ids)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(policy.call_index, 0)

    def test_context_limit_does_not_silently_trim(self):
        policy = self.make_policy()
        policy.config['max_context_tokens'] = 4
        with self.assertRaises(ContextLimitError): policy.act(AgentState('Test question', [], ACTION_SCHEMA))


if __name__ == '__main__': unittest.main()
