"""Actual tiny Qwen2 + PEFT LoRA CPU checks; no downloaded model weights."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('peft', reason='Install requirements-training.txt for real LoRA CPU tests')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.grpo_v1.synthetic import tiny_lora_model, tiny_base_model, synthetic_group, SyntheticTokenizer
from src.training.grpo_v1.loss import make_segment, generated_logprobs, clipped_segment_loss, group_advantages
from src.training.grpo_v1.trainer import GRPOTrainer, base_digest, save_checkpoint, reload_adapter_in_place
from src.training.grpo_v1.policy import assert_adapter_only, parameter_counts

CONFIG = json.loads((ROOT / 'configs/grpo_v1_smoke.json').read_text())


@pytest.fixture
def model():
    torch.set_num_threads(2)
    return tiny_lora_model()


def test_selective_logits_match_full_logits_and_gradients(model):
    segment = make_segment([], [1, 7, 8, 9], [10, 11, 2], 'sample')
    ids = torch.tensor([segment['prompt_token_ids'] + segment['generated_token_ids']])
    logits = model(input_ids=ids, use_cache=False).logits
    logits.retain_grad()
    mask = torch.tensor(segment['generated_token_mask'][1:])
    full_losses = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction='none')
    full_loss = (full_losses * mask).sum()
    selective = generated_logprobs(model, segment)
    torch.testing.assert_close(selective, -full_losses[mask.bool()])
    full_loss.backward()
    expected_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert torch.count_nonzero(logits.grad[0, :3]) == 0  # Prompt targets have exactly zero loss gradient.
    assert torch.count_nonzero(logits.grad[0, 3:6]) > 0
    model.zero_grad()
    (-generated_logprobs(model, segment).sum()).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, expected_grads[name], rtol=1e-5, atol=1e-6)


def test_group_update_changes_only_adapters_and_ratio_initially_one(model):
    group = synthetic_group(model)
    before_base = base_digest(model)
    before = {n: p.clone() for n, p in model.named_parameters()}
    for row in group:
        for segment in row['segments']:
            torch.testing.assert_close((generated_logprobs(model, segment) - torch.tensor(segment['old_logprobs'])).exp(),
                                       torch.ones(len(segment['generated_token_ids'])))
    trainer = GRPOTrainer(model, CONFIG['training'])
    result = trainer.update(group)
    assert result['update_applied'] and result['adapter_delta_l2'] > 0 and result['grad_norm'] > 0
    assert before_base == base_digest(model)
    changed = []
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter, before[name]):
            changed.append(name)
            assert '.lora_A.' in name or '.lora_B.' in name
        if '.lora_' not in name:
            assert not parameter.requires_grad and parameter.grad is None
    assert changed
    counts = parameter_counts(model)
    assert 0 < counts['trainable_params'] < counts['total_params']
    assert 0 < counts['trainable_ratio'] < 1


def test_all_equal_group_does_not_apply_adam_momentum(model):
    trainer = GRPOTrainer(model, CONFIG['training'])
    trainer.update(synthetic_group(model))
    before = {n: p.clone() for n, p in model.named_parameters()}
    for value in (0.0, 1.0):
        group = synthetic_group(model)
        for row in group:
            row['reward'] = value
        result = trainer.update(group)
        assert result['step'] == 1 and result['advantages'] == [0.0] * 4
        assert not result['update_applied'] and result['skip_reason'] == 'all_equal_rewards'
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_segment_backward_equals_whole_trajectory_gradient(model):
    group = synthetic_group(model)
    advantages = group_advantages([r['reward'] for r in group])
    combined = 0
    for row, advantage in zip(group, advantages):
        total = sum(len(s['generated_token_ids']) for s in row['segments'])
        for segment in row['segments']:
            combined = combined + clipped_segment_loss(generated_logprobs(model, segment),
                segment['old_logprobs'], advantage, total, 4, 0.2)
    combined.backward()
    expected = {n: p.grad.clone() for n, p in assert_adapter_only(model)}
    model.zero_grad()
    for row, advantage in zip(group, advantages):
        total = sum(len(s['generated_token_ids']) for s in row['segments'])
        for segment in row['segments']:
            clipped_segment_loss(generated_logprobs(model, segment), segment['old_logprobs'],
                                 advantage, total, 4, 0.2).backward()
    for name, p in assert_adapter_only(model):
        torch.testing.assert_close(p.grad, expected[name], rtol=1e-5, atol=1e-6)


def test_checkpoint_reload_in_place_and_fresh_peft_model(model, tmp_path):
    from peft import PeftModel
    trainer = GRPOTrainer(model, CONFIG['training'])
    group = synthetic_group(model)
    trainer.update(group)
    policy = SimpleNamespace(model=model, tokenizer=SyntheticTokenizer())
    checkpoint = tmp_path / 'checkpoints' / 'adapter'
    save_checkpoint(policy, trainer, checkpoint, dict(test=True))
    expected = generated_logprobs(model, group[0]['segments'][0]).detach()
    with torch.no_grad():
        for _, p in assert_adapter_only(model):
            p.add_(1)
    assert not torch.allclose(generated_logprobs(model, group[0]['segments'][0]), expected)
    reload_adapter_in_place(policy, checkpoint)
    torch.testing.assert_close(generated_logprobs(model, group[0]['segments'][0]), expected)
    fresh = tiny_base_model()
    loaded = PeftModel.from_pretrained(fresh, checkpoint, is_trainable=True).eval()
    torch.testing.assert_close(generated_logprobs(loaded, group[0]['segments'][0]), expected)
    state = torch.load(checkpoint / 'training_state.pt', weights_only=True)
    assert state['step'] == 1 and state['optimizer']['state']
    assert (checkpoint / 'adapter_model.safetensors').exists()
    assert not (checkpoint / 'model.safetensors').exists()


def test_update_budget_base_guard_and_nonfinite_rejection(model):
    trainer = GRPOTrainer(model, dict(CONFIG['training'], max_updates=1))
    trainer.update(synthetic_group(model))
    before = {n: p.clone() for n, p in model.named_parameters()}
    result = trainer.update(synthetic_group(model))
    assert result['skip_reason'] == 'update_budget_exhausted'
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    next(p for n, p in model.named_parameters() if '.lora_' not in n).requires_grad_(True)
    with pytest.raises(ValueError, match='base must be frozen'):
        assert_adapter_only(model)
    with pytest.raises(ValueError, match='logprob'):
        clipped_segment_loss(torch.tensor([-1.], requires_grad=True), [float('nan')], 1., 1, 4, 0.2)


def test_clipping_math_and_denominator():
    current = torch.log(torch.tensor([1.5, 0.5], requires_grad=True))
    old = [0., 0.]
    positive = clipped_segment_loss(current, old, 1., 10, 4, 0.2)
    negative = clipped_segment_loss(current, old, -1., 10, 4, 0.2)
    assert positive.item() == pytest.approx(-(1.2 + 0.5) / 40)
    assert negative.item() == pytest.approx((1.5 + 0.8) / 40)


def test_training_policy_captures_actual_frozen_act_calls(model):
    from transformers import BatchEncoding
    from unittest.mock import patch
    from src.training.grpo_v1.policy import TrainingPolicy
    from src.training.grpo_v1.collector import collect_group
    from src.env_v1 import generate_kb
    from src.env_v1.tasks import generate_tasks
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return json.dumps(messages)
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            self.text = text
            ids = [3 + sum(map(ord, word)) % 60 for word in text.split()]
            return (BatchEncoding(dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)))
                    if return_tensors else dict(input_ids=ids))
        def decode(self, ids, skip_special_tokens=True):
            return ('{"action":"lookup","arguments":{"entity":"s00"}}' if ids[0] == 12 else
                    '{"action":"direct_answer","arguments":{"final_answer":3.8}}')
    tokenizer = Tokenizer()
    model.config.max_position_embeddings = 16384
    policy = TrainingPolicy(model, tokenizer, CONFIG)
    calls = []
    def scripted_generate(**kwargs):
        calls.append(kwargs)
        assert kwargs['do_sample'] and kwargs['temperature'] == 0.8 and kwargs['top_p'] == 0.95
        next_id = 13 if 'observation' in tokenizer.text and '3.8' in tokenizer.text else 12
        return torch.cat((kwargs['input_ids'], torch.tensor([[next_id, 2]])), dim=1)
    kb = generate_kb()
    rng_before = torch.get_rng_state().clone()
    with patch.object(model, 'generate', side_effect=scripted_generate):
        group = collect_group(dict(kb=kb, task=generate_tasks(kb)[0], seed=42), policy, CONFIG, 'actual-act')
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert len(calls) == 8
    assert [r['reward'] for r in group] == [1] * 4
    for row in group:
        assert [s['generated_token_ids'] for s in row['segments']] == [[12, 2], [13, 2]]
        for segment in row['segments']:
            assert len(segment['old_logprobs']) == 2
            assert sum(segment['generated_token_mask']) == 2
            with torch.no_grad():
                torch.testing.assert_close(generated_logprobs(model, segment), torch.tensor(segment['old_logprobs']))
