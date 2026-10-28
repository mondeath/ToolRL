"""Dynamic one-pass V2 runner plus post-run correctness and artifact audits."""
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
import time

from src.training.grpo_v1.baseline_reporting import group_metrics, training_summary
from src.training.grpo_v1.baseline_runner import schedule, write_json
from src.training.grpo_v1.data import read_json
from .data import load_v2_data, dry_run, load_v2_smoke_data, smoke_dry_run
from .reporting import render_report, write_manifest, validate_artifact_manifest


def validate_checkpoint(config, checkpoint, split):
    metadata = read_json(Path(checkpoint) / 'metadata.json')
    if metadata['config'] != config or metadata['selection_sha256'] != config['tasks_sha256']:
        raise ValueError('V2 checkpoint/config/selection mismatch')
    if metadata['attempted_groups'] not in config['checkpoint_groups']:
        raise ValueError('Not a scheduled V2 checkpoint')
    if split == 'diagnostic' and metadata['attempted_groups'] != config['task_count']:
        raise ValueError('Diagnostic evaluation is only allowed on the final V2 checkpoint')
    return metadata


def audit_training_outputs(directory, config):
    directory = Path(directory)
    metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line]
    rollouts = [json.loads(line) for line in (directory / 'rollouts.jsonl').read_text().splitlines() if line]
    if len(metrics) != config['task_count'] or len(rollouts) != config['task_count'] * config['group_size']:
        raise ValueError('Incomplete training metrics or rollouts')
    expected_step = 0
    ratio_errors, clip_fractions, approx_kls, mixed = [], [], [], 0
    for row in metrics:
        is_mixed = bool(row['mixed_reward_group'])
        if row['zero_variance_group'] == is_mixed:
            raise ValueError('Reward variance/mixed-group classification mismatch')
        if bool(row['optimizer_step_performed']) != is_mixed:
            raise ValueError('Mixed groups must update and zero-variance groups must skip')
        expected_step += int(is_mixed)
        if row['optimizer_steps'] != expected_step:
            raise ValueError('Optimizer step accounting mismatch')
        if is_mixed:
            mixed += 1
            if row['reward_variance'] <= 0 or row['adapter_delta_l2'] <= 0:
                raise ValueError('Mixed group lacks reward variance or LoRA delta')
            if row['ratio_token_count'] <= 0:
                raise ValueError('Mixed group lacks ratio diagnostics')
            ratio_errors.append(max(abs(row['ratio_min'] - 1), abs(row['ratio_mean'] - 1),
                                    abs(row['ratio_max'] - 1)))
            clip_fractions.append(row['clip_fraction'])
            approx_kls.append(row['approx_kl'])
        elif row['reward_variance'] != 0 or row['adapter_delta_l2'] != 0:
            raise ValueError('Zero-variance group changed adapters')
    if ratio_errors and (max(ratio_errors) > 1e-4 or max(clip_fractions) != 0 or max(approx_kls) > 1e-7):
        raise ValueError('Initial old/current log-prob alignment diagnostics failed')
    for rollout in rollouts:
        for segment in rollout['segments']:
            old = segment['old_logprobs']
            if len(old) != len(segment['generated_token_ids']) or not all(math.isfinite(value) for value in old):
                raise ValueError('Stored old log-probs are incomplete or non-finite')
    final_group = config['task_count']
    final_metadata = read_json(directory / f'checkpoint_group_{final_group:03}.json')
    if not final_metadata.get('checkpoint_reload') or final_metadata.get('max_reload_logprob_error', 1) > 1e-4:
        raise ValueError('Final adapter checkpoint reload validation missing or inaccurate')
    required = [directory / 'config.json', directory / 'tasks.json', directory / 'rollouts.jsonl',
        directory / 'metrics.jsonl', directory / 'training_summary.json', directory / 'status.json',
        directory / 'base_dev/results.jsonl', directory / f'eval_group_{final_group:03}/dev/results.jsonl',
        directory / f'eval_group_{final_group:03}/diagnostic/results.jsonl',
        directory / 'checkpoints' / f'group_{final_group:03}' / 'training_state.pt']
    missing = [str(path.relative_to(directory)) for path in required if not path.is_file()]
    if missing:
        raise ValueError('Missing required run artifacts: ' + ', '.join(missing))
    return dict(status='passed', attempted_groups=len(metrics), optimizer_steps=expected_step,
        mixed_reward_groups=mixed, zero_variance_groups=len(metrics) - mixed,
        max_abs_ratio_error=max(ratio_errors, default=None), max_clip_fraction=max(clip_fractions, default=None),
        max_approx_kl=max(approx_kls, default=None), positive_lora_delta_groups=mixed,
        checkpoint_reload=True, old_logprobs_complete=True, output_files_complete=True)


def _train(root, config, snapshot, directory, checks, data_loader, dry_run_loader):
    import torch
    from src.training.grpo_v1.collector import collect_group
    from src.training.grpo_v1.evaluation import greedy_eval
    from src.training.grpo_v1.loss import generated_logprobs
    from src.training.grpo_v1.policy import load_policy, parameter_counts
    from src.training.grpo_v1.trainer import (GRPOTrainer, base_digest, memory_stats,
        reload_adapter_in_place, save_checkpoint)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train, dev, diagnostic = data_loader(root, config)
    write_json(directory / 'config.json', config)
    write_json(directory / 'tasks.json', read_json(Path(root) / config['tasks_file']))
    write_json(directory / 'dry_run.json', dry_run_loader(root, config))
    write_json(directory / 'preflight.json', checks)
    (directory / 'runtime_requirements.txt').write_text('\n'.join(sorted(
        f'{distribution.metadata["Name"]}=={distribution.version}'
        for distribution in importlib.metadata.distributions())) + '\n')
    groups, checkpoints = [], []
    write_json(directory / 'status.json', dict(status='loading', training_started=False))
    try:
        torch.cuda.reset_peak_memory_stats(0)
        policy = load_policy(snapshot, config)
        write_json(directory / 'model_runtime.json', {**policy.runtime_metadata,
            **parameter_counts(policy.model), **memory_stats()})
        base_hash = base_digest(policy.model)
        trainer = GRPOTrainer(policy.model, config['training'])
        probe = None

        def collect(item, attempted):
            nonlocal probe
            group = collect_group(item, policy, config, group_id=f'v2-{attempted:03}')
            if probe is None:
                probe = deepcopy(group[0]['segments'][0])
            for row in group:
                rollout_stream.write(json.dumps(row, allow_nan=False) + '\n')
            rollout_stream.flush()
            return group

        def save(attempted):
            checkpoint = directory / 'checkpoints' / f'group_{attempted:03}'
            metadata = dict(config=config, selection_sha256=config['tasks_sha256'],
                attempted_groups=attempted, optimizer_steps=trainer.step, base_digest=base_hash,
                snapshot=str(snapshot), kind='initialized_zero-output_LoRA_base' if attempted == 0 else 'success_only_GRPO_V2')
            save_checkpoint(policy, trainer, checkpoint, metadata)
            policy.model.eval()
            if probe is not None:
                with torch.no_grad():
                    expected = generated_logprobs(policy.model, probe).cpu()
                reload_adapter_in_place(policy, checkpoint)
                with torch.no_grad():
                    actual = generated_logprobs(policy.model, probe).cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                metadata['checkpoint_reload'] = True
                metadata['max_reload_logprob_error'] = float((actual - expected).abs().max())
            write_json(directory / f'checkpoint_group_{attempted:03}.json', metadata)
            checkpoints.append(attempted)

        def evaluate(items, attempted, split):
            if split == 'diagnostic' and attempted != config['task_count']:
                raise AssertionError('Diagnostic cannot run before final checkpoint')
            target = directory / ('base_dev' if attempted == 0 else f'eval_group_{attempted:03}/{split}')
            if attempted == 0:
                with policy.model.disable_adapter():
                    result = greedy_eval(items, policy, config, target)
            else:
                result = greedy_eval(items, policy, config, target)
            result.update(attempted_groups=attempted, optimizer_steps=trainer.step,
                          wall_clock_seconds=time.monotonic() - started)
            write_json(target / 'summary.json', result)
            render_report(directory)

        def log(group, result, attempted):
            metric = group_metrics(group, result, attempted, time.monotonic() - started, memory_stats())
            if bool(metric['mixed_reward_group']) != bool(metric['optimizer_step_performed']):
                raise AssertionError('Mixed groups must update and equal groups must skip')
            groups.append(metric)
            metrics_stream.write(json.dumps(metric, allow_nan=False) + '\n')
            metrics_stream.flush()
            write_json(directory / 'training_summary.json', training_summary(groups))
            write_json(directory / 'status.json', dict(status='training', training_started=True,
                attempted_groups=attempted, optimizer_steps=trainer.step,
                wall_clock_seconds=time.monotonic() - started, **memory_stats()))

        with (directory / 'rollouts.jsonl').open('w') as rollout_stream, \
             (directory / 'metrics.jsonl').open('w') as metrics_stream:
            schedule(train, dev, diagnostic, config, collect, trainer.update, save, evaluate, log)
        if base_hash != base_digest(policy.model):
            raise AssertionError('Base weights changed')
        summary = training_summary(groups)
        if trainer.step != summary['mixed_group_count']:
            raise AssertionError('Optimizer count does not match mixed groups')
        status = dict(status='completed', training_started=True, attempted_groups=len(train),
            optimizer_steps=trainer.step, checkpoints=checkpoints, base_unchanged=True,
            adapter_changed=any(row['adapter_delta_l2'] > 0 for row in groups), checkpoint_reload=True,
            no_effective_update=not any(row['adapter_delta_l2'] > 0 for row in groups),
            wall_clock_seconds=time.monotonic() - started, **memory_stats())
        write_json(directory / 'status.json', status)
        audit = audit_training_outputs(directory, config)
        write_json(directory / 'correctness_audit.json', audit)
        render_report(directory)
        return dict(**status, correctness_audit=audit)
    except Exception as exc:
        write_json(directory / 'status.json', dict(status='failed', error_type=type(exc).__name__,
            error=str(exc), attempted_groups=len(groups), wall_clock_seconds=time.monotonic() - started,
            settings_automatically_changed=False, **memory_stats()))
        render_report(directory)
        raise


def _run_with_manifest(root, config, snapshot, directory, checks, data_loader, dry_run_loader):
    train, _, _ = data_loader(root, config)
    try:
        result = _train(root, config, snapshot, directory, checks, data_loader, dry_run_loader)
    finally:
        if Path(directory).is_dir():
            write_manifest(directory, config, train)
    if result['status'] == 'completed':
        result['artifact_manifest'] = validate_artifact_manifest(directory)
    return result


def train_v2(root, config, snapshot, directory, checks):
    return _run_with_manifest(root, config, snapshot, directory, checks, load_v2_data, dry_run)


def train_v2_smoke(root, config, snapshot, directory, checks):
    return _run_with_manifest(root, config, snapshot, directory, checks,
                              load_v2_smoke_data, smoke_dry_run)
