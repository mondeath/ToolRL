"""Per-generation alignment and whole-trajectory GRPO normalization."""
from contextlib import nullcontext
import math
import torch
import torch.nn.functional as F


def make_segment(messages, prompt_ids, generated_ids, raw_output, **metadata):
    if not prompt_ids or not generated_ids:
        raise ValueError('A generation must have nonempty prompt and completion')
    return dict(messages=messages, prompt_token_ids=list(prompt_ids), generated_token_ids=list(generated_ids),
                generated_token_mask=[0] * len(prompt_ids) + [1] * len(generated_ids),
                raw_output=raw_output, **metadata)


def validate_segment(segment):
    p, g = segment['prompt_token_ids'], segment['generated_token_ids']
    if not p or not g or segment['generated_token_mask'] != [0] * len(p) + [1] * len(g):
        raise ValueError('Invalid generated-token mask; all context must be masked')
    if any(type(token) is not int or token < 0 for token in p + g):
        raise ValueError('Invalid token ID')


def group_advantages(rewards):
    if not rewards or any(r not in (0, 1) or not math.isfinite(r) for r in rewards):
        raise ValueError('Only binary evaluator success rewards are supported')
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return [0.0] * len(rewards) if variance == 0 else [(r - mean) / (math.sqrt(variance) + 1e-8) for r in rewards]


def group_rates(reward_groups):
    if not reward_groups:
        raise ValueError('No groups')
    for rewards in reward_groups:
        group_advantages(rewards)
    n = len(reward_groups)
    return dict(mixed_reward_group_rate=sum(min(r) != max(r) for r in reward_groups) / n,
                all_success_group_rate=sum(all(r) for r in reward_groups) / n,
                all_failure_group_rate=sum(not any(r) for r in reward_groups) / n)


def generated_logprobs(model, segment):
    """Qwen2 logits_to_keep projects only positions predicting generated tokens.

    The full prompt still participates in attention. Never compute a loss on
    observations, system/user text, or copied prior action text in that prompt.
    """
    validate_segment(segment)
    device = next(model.parameters()).device
    prompt, generated = segment['prompt_token_ids'], segment['generated_token_ids']
    ids = torch.tensor([prompt + generated], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt) - 1, ids.shape[1] - 1, device=device)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with context:
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=positions).logits
    if logits.shape[:2] != (1, len(generated)):
        raise ValueError('Model must implement selective logits_to_keep')
    targets = ids[:, len(prompt):].reshape(-1)
    result = -F.cross_entropy(logits[0].float(), targets, reduction='none')
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite generated logprobs')
    return result


def clipped_segment_loss(current, old, advantage, total_trajectory_tokens, group_size, epsilon):
    if total_trajectory_tokens < current.numel() or total_trajectory_tokens < 1 or group_size < 1:
        raise ValueError('Invalid whole-trajectory normalization')
    old = torch.as_tensor(old, dtype=current.dtype, device=current.device).detach()
    if old.shape != current.shape or not torch.isfinite(old).all():
        raise ValueError('Old logprob alignment/values invalid')
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    loss = -surrogate.sum() / (group_size * total_trajectory_tokens)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite GRPO loss')
    return loss


def ratio_diagnostics(current, old, epsilon):
    """Return detached PPO/GRPO diagnostics in the raw-token log-prob space.

    ``approx_kl`` uses the non-negative k3 estimator
    ``(ratio - 1) - log(ratio)``.  Callers aggregate these token-level sums
    before dividing, so trajectories with multiple assistant turns retain the
    same whole-trajectory accounting as the loss.
    """
    if not 0 < epsilon < 1:
        raise ValueError('clip epsilon must be in (0, 1)')
    old = torch.as_tensor(old, dtype=current.dtype, device=current.device).detach()
    if old.shape != current.shape or not torch.isfinite(old).all() or not torch.isfinite(current).all():
        raise ValueError('Old/current logprob alignment/values invalid')
    with torch.no_grad():
        log_ratio = current.detach() - old
        ratio = log_ratio.exp()
        if not torch.isfinite(ratio).all():
            raise FloatingPointError('Non-finite importance ratio')
        clipped = (ratio < 1 - epsilon) | (ratio > 1 + epsilon)
        approx_kl = (ratio - 1) - log_ratio
        return dict(token_count=ratio.numel(), ratio_sum=float(ratio.sum()),
                    ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                    clipped_token_count=int(clipped.sum()), approx_kl_sum=float(approx_kl.sum()))
