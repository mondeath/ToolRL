"""One fixed pass, quarter checkpoints, dev only until final diagnostic."""
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import time

from .data import read_json
from .baseline_data import load_baseline_data, dry_run
from .baseline_reporting import group_metrics, training_summary, render_report


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def schedule(train, dev, diagnostic, config, collect, update, save, evaluate, log):
    """Testable schedule; no best-checkpoint selection or repeated diagnostic."""
    save(0)
    evaluate(dev, 0, 'dev')
    for attempted, item in enumerate(train, 1):
        group = collect(item, attempted)
        result = update(group)
        log(group, result, attempted)
        if attempted in config['checkpoint_groups'][1:]:
            save(attempted)
            evaluate(dev, attempted, 'dev')
    evaluate(diagnostic, len(train), 'diagnostic')


def validate_checkpoint(root, config, checkpoint, split):
    metadata = read_json(Path(checkpoint) / 'metadata.json')
    if metadata['config'] != config or metadata['selection_sha256'] != config['tasks_sha256']:
        raise ValueError('Checkpoint/config/selection mismatch')
    if metadata['attempted_groups'] not in config['checkpoint_groups']:
        raise ValueError('Not a scheduled baseline checkpoint')
    if split == 'diagnostic' and metadata['attempted_groups'] != 240:
        raise ValueError('Diagnostic evaluation is only allowed on the final checkpoint')
    return metadata


def import_base_diagnostic(smoke_dir, config, diagnostic):
    """Reuse an existing NF4 smoke baseline, never initiate another diagnostic."""
    from .evaluation import metrics
    smoke_dir = Path(smoke_dir)
    original = read_json(smoke_dir / 'config.json')
    for key in ('model_name', 'revision', 'quantization', 'lora', 'rollout', 'sampling', 'diagnostic_tasks_sha256'):
        if original[key] != config[key]:
            raise ValueError('Archived NF4 smoke baseline config mismatch: ' + key)
    base_dir = smoke_dir / 'before_nf4'
    summary = read_json(base_dir / 'summary.json')
    rows = [json.loads(line) for line in (base_dir / 'results.jsonl').read_text().splitlines() if line.strip()]
    if (summary['do_sample'] or summary['task_ids'] != [x['task']['task_id'] for x in diagnostic]
            or [row['task_id'] for row in rows] != summary['task_ids']):
        raise ValueError('Archived baseline is not the fixed greedy diagnostic')
    import hashlib
    return dict(**metrics(rows), provenance=dict(source=str(base_dir),
        sha256=hashlib.sha256((base_dir / 'results.jsonl').read_bytes()).hexdigest(),
        note='Existing smoke adapter-off NF4 artifact; not rerun or used for checkpoint selection'))


def train_baseline(root, config, snapshot, directory, checks, base_smoke_dir=None):
    import torch
    from .policy import load_policy, parameter_counts
    from .collector import collect_group
    from .trainer import GRPOTrainer, base_digest, save_checkpoint, reload_adapter_in_place, memory_stats
    from .loss import generated_logprobs
    from .evaluation import greedy_eval

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train, dev, diagnostic = load_baseline_data(root, config)
    write_json(directory / 'config.json', config)
    write_json(directory / 'tasks.json', read_json(Path(root) / config['tasks_file']))
    write_json(directory / 'dry_run.json', dry_run(root, config))
    write_json(directory / 'preflight.json', checks)
    (directory / 'runtime_requirements.txt').write_text('\n'.join(sorted(
        f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())) + '\n')
    groups, checkpoints = [], []
    write_json(directory / 'status.json', dict(status='loading', training_started=False))
    try:
        if base_smoke_dir:
            write_json(directory / 'base_nf4_diagnostic.json', import_base_diagnostic(base_smoke_dir, config, diagnostic))
        torch.cuda.reset_peak_memory_stats(0)
        policy = load_policy(snapshot, config)
        write_json(directory / 'model_runtime.json', {**policy.runtime_metadata, **parameter_counts(policy.model), **memory_stats()})
        base_hash = base_digest(policy.model)
        trainer = GRPOTrainer(policy.model, config['training'])
        probe = None
        def collect(item, attempted):
            nonlocal probe
            group = collect_group(item, policy, config, group_id=f'baseline-{attempted:03}')
            if probe is None:
                probe = deepcopy(group[0]['segments'][0])
            for row in group:
                rollout_stream.write(json.dumps(row, allow_nan=False) + '\n')
            rollout_stream.flush()
            return group
        def save(attempted):
            checkpoint = directory / 'checkpoints' / f'group_{attempted:03}'
            metadata = dict(config=config, selection_sha256=config['tasks_sha256'], attempted_groups=attempted,
                optimizer_steps=trainer.step, base_digest=base_hash, snapshot=str(snapshot),
                kind='initialized_zero-output_LoRA_base' if attempted == 0 else 'task_only_GRPO')
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
            if split == 'diagnostic' and attempted != 240:
                raise AssertionError('Diagnostic cannot run mid-training')
            if attempted == 0:
                with policy.model.disable_adapter():
                    result = greedy_eval(items, policy, config, directory / 'base_dev')
            else:
                result = greedy_eval(items, policy, config, directory / f'eval_group_{attempted:03}' / split)
            result.update(attempted_groups=attempted, optimizer_steps=trainer.step,
                          wall_clock_seconds=time.monotonic() - started)
            write_json(directory / ('base_dev/summary.json' if attempted == 0 else f'eval_group_{attempted:03}/{split}/summary.json'), result)
            render_report(directory)
        def log(group, result, attempted):
            metric = group_metrics(group, result, attempted, time.monotonic() - started, memory_stats())
            if bool(metric['mixed_reward_group']) != bool(metric['optimizer_step_performed']):
                raise AssertionError('Every mixed group must update, and equal groups must not')
            groups.append(metric)
            metrics_stream.write(json.dumps(metric, allow_nan=False) + '\n')
            metrics_stream.flush()
            summary = training_summary(groups)
            write_json(directory / 'training_summary.json', summary)
            write_json(directory / 'status.json', dict(status='training', training_started=True,
                attempted_groups=attempted, optimizer_steps=trainer.step, wall_clock_seconds=time.monotonic() - started, **memory_stats()))
            print(json.dumps(dict(task_id=metric['task_id'], difficulty=metric['difficulty'],
                rewards=metric['rewards'], attempted_groups=attempted, optimizer_steps=trainer.step,
                mixed_reward_group_rate=summary['mixed_reward_group_rate'], elapsed_seconds=metric['elapsed_seconds'])), flush=True)
        with (directory / 'rollouts.jsonl').open('w') as rollout_stream, (directory / 'metrics.jsonl').open('w') as metrics_stream:
            schedule(train, dev, diagnostic, config, collect, trainer.update, save, evaluate, log)
        if base_hash != base_digest(policy.model):
            raise AssertionError('Base weights changed')
        if trainer.step != training_summary(groups)['mixed_group_count']:
            raise AssertionError('Optimizer count does not match mixed groups')
        status = dict(status='completed', training_started=True, attempted_groups=240, optimizer_steps=trainer.step,
            checkpoints=checkpoints, base_unchanged=True, adapter_changed=any(row['adapter_delta_l2'] > 0 for row in groups),
            checkpoint_reload=True, no_effective_update=not any(row['adapter_delta_l2'] > 0 for row in groups),
            wall_clock_seconds=time.monotonic() - started, **memory_stats())
        write_json(directory / 'status.json', status)
    except Exception as exc:
        status = dict(status='failed', error_type=type(exc).__name__, error=str(exc),
            attempted_groups=len(groups), wall_clock_seconds=time.monotonic() - started,
            settings_automatically_changed=False, **memory_stats())
        write_json(directory / 'status.json', status)
        render_report(directory)
        raise
    render_report(directory)
    return status
