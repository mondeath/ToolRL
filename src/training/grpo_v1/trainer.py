"""One on-policy update per mixed group; optional smoke-only update budget."""
import hashlib
import json
from pathlib import Path
import torch

from .loss import generated_logprobs, group_advantages, clipped_segment_loss
from .policy import assert_adapter_only


def memory_stats():
    if not torch.cuda.is_available():
        return dict(gpu_allocated=None, gpu_reserved=None, gpu_peak_memory=None, gpu_peak_reserved=None)
    return dict(gpu_allocated=torch.cuda.memory_allocated(0), gpu_reserved=torch.cuda.memory_reserved(0),
                gpu_peak_memory=torch.cuda.max_memory_allocated(0), gpu_peak_reserved=torch.cuda.max_memory_reserved(0))


def base_digest(model):
    """Streaming packed base hash before/after training; no second 7B copy."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            continue
        digest.update(name.encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 1024 * 1024):
            digest.update(flat[start:start + 1024 * 1024].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class GRPOTrainer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.named_adapters = assert_adapter_only(model)
        self.optimizer = torch.optim.AdamW([p for _, p in self.named_adapters],
            lr=config['learning_rate'], weight_decay=0.0)
        self.step = 0

    def update(self, group):
        assert_adapter_only(self.model)
        if len(group) != 4 or len({r['task_id'] for r in group}) != 1:
            raise ValueError('One update requires four trajectories of the same task')
        advantages = group_advantages([row['reward'] for row in group])
        result = dict(step=self.step, policy_loss=0.0, kl=None, grad_norm=0.0,
                      learning_rate=self.optimizer.param_groups[0]['lr'], advantages=advantages,
                      adapter_delta_l2=0.0, update_applied=False)
        if not any(advantages):
            result['skip_reason'] = 'all_equal_rewards'
            return result  # No Adam momentum/weight-decay step on zero signal.
        if self.config['max_updates'] is not None and self.step >= self.config['max_updates']:
            result['skip_reason'] = 'update_budget_exhausted'
            return result
        before = {name: p.detach().cpu().clone() for name, p in self.named_adapters}
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()  # checkpointing on, dropout disabled at load time
        loss_value = 0.0
        for row, advantage in zip(group, advantages):
            total = sum(len(s['generated_token_ids']) for s in row['segments'])
            if total == 0:
                raise ValueError('Empty trajectory')
            for segment in row['segments']:
                current = generated_logprobs(self.model, segment)
                loss = clipped_segment_loss(current, segment['old_logprobs'], advantage,
                    total, len(group), self.config['clip_epsilon'])
                loss.backward()
                loss_value += loss.detach().item()
                del current, loss  # Free this turn's graph before replaying next turn.
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.named_adapters],
            self.config['max_grad_norm'], error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        delta = sum((p.detach().cpu() - before[name]).float().square().sum().item() for name, p in self.named_adapters) ** 0.5
        self.model.eval()
        result.update(step=self.step, policy_loss=loss_value, grad_norm=float(norm),
                      adapter_delta_l2=delta, update_applied=True)
        return result


def save_checkpoint(policy, trainer, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    assert_adapter_only(policy.model)
    # PEFT save_pretrained saves adapters only; never merge or save the 7B base.
    policy.model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    policy.tokenizer.save_pretrained(directory)
    torch.save(dict(optimizer=trainer.optimizer.state_dict(), step=trainer.step,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
               directory / 'training_state.pt')
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def reload_adapter_in_place(policy, directory):
    """Reload adapter tensors onto the same frozen base; no second model."""
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    weights = load_peft_weights(str(directory), device='cpu', local_files_only=True)
    incompatible = set_peft_model_state_dict(policy.model, weights, adapter_name='default')
    if incompatible.unexpected_keys or any('lora_' in key for key in incompatible.missing_keys):
        raise ValueError('Adapter checkpoint key mismatch')
    assert_adapter_only(policy.model)
