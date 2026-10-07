"""Explicit tiny CPU-only numerical smoke. Never used as the task policy."""
import json
from types import SimpleNamespace
import torch

from .loss import make_segment, generated_logprobs
from .policy import parameter_counts
from .trainer import GRPOTrainer, base_digest, save_checkpoint, reload_adapter_in_place


def tiny_base_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(2026)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0, eos_token_id=2, pad_token_id=0))


def tiny_lora_model():
    from peft import LoraConfig, get_peft_model
    base = tiny_base_model()
    return get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=8,
                                         lora_dropout=0.0, target_modules=['q_proj', 'v_proj'])).eval()


def synthetic_group(model):
    group = []
    for index in range(4):
        segments = [make_segment([], [1, 7, 8], [10 + index, 11 + index, 2], 'synthetic'),
                    make_segment([], [1, 7, 8, 9, 10 + index], [20 + index, 2], 'synthetic')]
        with torch.no_grad():
            for segment in segments:
                segment['old_logprobs'] = generated_logprobs(model, segment).tolist()
        group.append(dict(task_id='synthetic-only', reward=float(index % 2 == 0), segments=segments))
    return group


class SyntheticTokenizer:
    def save_pretrained(self, directory):
        (directory / 'synthetic_tokenizer.json').write_text('{"synthetic_only": true}\n')


def cpu_smoke(config, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = tiny_lora_model()
    before = base_digest(model)
    group = synthetic_group(model)
    trainer = GRPOTrainer(model, config['training'])
    update = trainer.update(group)
    if update['adapter_delta_l2'] <= 0 or before != base_digest(model):
        raise AssertionError('Adapter update/frozen-base check failed')
    policy = SimpleNamespace(model=model, tokenizer=SyntheticTokenizer())
    checkpoint = output / 'checkpoints' / 'adapter'
    save_checkpoint(policy, trainer, checkpoint, dict(synthetic_only=True))
    with torch.no_grad():
        expected = generated_logprobs(model, group[0]['segments'][0]).clone()
        # Corrupt adapters to prove reload restores them, not a no-op check.
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.add_(0.1)
    reload_adapter_in_place(policy, checkpoint)
    with torch.no_grad():
        actual = generated_logprobs(model, group[0]['segments'][0])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    result = dict(status='passed', synthetic_only=True, device='cpu', update=update,
                  base_unchanged=True, checkpoint_reload=True, **parameter_counts(model))
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
