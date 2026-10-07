"""Explicit static/preflight/CPU checks or bounded QLoRA smoke; no default training."""
from _common import ROOT
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from src.training.grpo_v1.data import read_json, load_data
from src.training.grpo_v1.preflight import preflight


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def archived_baseline(root, config, diagnostic):
    from src.training.grpo_v1.evaluation import metrics
    from src.evaluation.model_scale import digest
    path = root / config['archived_base_results']
    summary_path = root / config['archived_base_summary']
    if not path.exists() or not summary_path.exists():
        raise ValueError('Frozen bf16 base results/summary missing; do not substitute NF4 results')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    summary = read_json(summary_path)
    run = summary['run']
    ids = [x['task']['task_id'] for x in diagnostic]
    if ([r['task_id'] for r in rows] != ids or run['selected_task_ids'] != ids
            or run['model_identity']['revision'] != config['revision']
            or run['model_config']['do_sample'] or run['model_config']['dtype'] != 'bfloat16'
            or run['kb_sha256'] != digest(diagnostic[0]['kb'])
            or run['tasks_sha256'] != digest([item['task'] for item in diagnostic])
            or run['rollout_config'] != config['rollout'] or run['failure_rate'] != 0):
        raise ValueError('Archived baseline contract mismatch')
    return dict(metrics=metrics(rows), provenance=dict(path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), run=run))


def run_training(config, snapshot, output, checks):
    import importlib.metadata
    import torch
    from src.training.grpo_v1.policy import load_policy, parameter_counts
    from src.training.grpo_v1.collector import collect_group
    from src.training.grpo_v1.loss import group_rates, generated_logprobs
    from src.training.grpo_v1.trainer import GRPOTrainer, base_digest, memory_stats, save_checkpoint, reload_adapter_in_place
    from src.training.grpo_v1.evaluation import greedy_eval

    items, diagnostic = load_data(ROOT, config)
    archive = archived_baseline(ROOT, config, diagnostic)
    write_json(output / 'archived_bf16_base.json', archive)
    write_json(output / 'config.json', config)
    write_json(output / 'smoke_tasks.json', read_json(ROOT / config['tasks_file']))
    write_json(output / 'preflight.json', checks)
    (output / 'runtime_requirements.txt').write_text('\n'.join(sorted(
        f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())) + '\n')
    write_json(output / 'status.json', dict(status='running', training_started=False))
    torch.cuda.reset_peak_memory_stats(0)
    policy = load_policy(snapshot, config)
    write_json(output / 'model_runtime.json', dict(**policy.runtime_metadata, **memory_stats()))
    print(json.dumps(parameter_counts(policy.model)), flush=True)
    base_before = base_digest(policy.model)
    # Baseline on the SAME quantized base, no adapters. Keep archived bf16 separately.
    with policy.model.disable_adapter():
        before = greedy_eval(diagnostic, policy, config, output / 'before_nf4')
    before_memory = memory_stats()
    torch.cuda.reset_peak_memory_stats(0)
    trainer = GRPOTrainer(policy.model, config['training'])
    rewards_history, adapter_changed = [], False
    probe = None
    with (output / 'rollouts.jsonl').open('w') as rollout_stream, (output / 'metrics.jsonl').open('w') as metrics_stream:
        # Always cover all 24 tasks once. After three updates, collect remaining
        # groups without stepping. No resampling for mixed rewards, no extra epochs.
        for attempted, item in enumerate(items, 1):
            group = collect_group(item, policy, config, group_id=f'group-{attempted:02}')
            for row in group:
                rollout_stream.write(json.dumps(row, allow_nan=False) + '\n')
            rollout_stream.flush()
            if probe is None:
                probe = deepcopy(group[0]['segments'][0])
            update = trainer.update(group)
            adapter_changed |= update['adapter_delta_l2'] > 0
            rewards_history.append([r['reward'] for r in group])
            mean = lambda key: sum(r[key] for r in group) / len(group)
            metric = dict(attempted_groups=attempted, task_id=item['task']['task_id'], **update,
                mean_reward=mean('reward'), success_rate=mean('success'),
                avg_trajectory_length=mean('trajectory_length'), avg_generated_tokens=mean('generated_token_count'),
                invalid_action_rate=sum(r['termination_reason'] == 'invalid_action' for r in group) / 4,
                max_tool_call_rate=sum(r['termination_reason'] == 'max_tool_calls' for r in group) / 4,
                avg_tool_cost=mean('tool_cost'), rule_violation_count=sum(len(r['rule_violations']) for r in group),
                **group_rates(rewards_history), **memory_stats())
            metrics_stream.write(json.dumps(metric, allow_nan=False) + '\n')
            metrics_stream.flush()
            print(json.dumps(metric, allow_nan=False), flush=True)
    train_memory = memory_stats()
    base_unchanged = base_before == base_digest(policy.model)
    if not base_unchanged:
        raise AssertionError('Frozen base weights changed')
    checkpoint = output / 'checkpoints' / 'adapter'
    metadata = dict(config=config, preflight=checks, base_digest=base_before,
                    optimizer_steps=trainer.step, tasks_sha256=config['tasks_sha256'])
    save_checkpoint(policy, trainer, checkpoint, metadata)
    policy.model.eval()
    with torch.no_grad():
        expected = generated_logprobs(policy.model, probe).cpu()
    reload_adapter_in_place(policy, checkpoint)
    with torch.no_grad():
        actual = generated_logprobs(policy.model, probe).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    write_json(output / 'checkpoint_reload.json', dict(passed=True, max_logprob_error=float((actual - expected).abs().max())))
    torch.cuda.reset_peak_memory_stats(0)
    after = greedy_eval(diagnostic, policy, config, output / 'after_nf4')
    after_memory = memory_stats()
    status = ('no_effective_update' if not adapter_changed or not trainer.step
              else 'passed' if after['has_legal_tool_actions'] else 'post_eval_no_legal_tool_actions')
    result = dict(status=status,
        archived_bf16_base=archive, before_nf4=before, after_nf4=after,
        attempted_groups=24, optimizer_steps=trainer.step, adapter_changed=adapter_changed,
        base_unchanged=base_unchanged, checkpoint_reload=True, **group_rates(rewards_history),
        memory=dict(before_eval=before_memory, training=train_memory, after_eval=after_memory),
        parameters=parameter_counts(policy.model), performance_conclusion=False)
    write_json(output / 'summary.json', result)
    write_json(output / 'status.json', dict(status=result['status'], training_started=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grpo_v1_smoke.json')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--static-check', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--cpu-smoke', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Saved PEFT checkpoint for --eval only')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.checkpoint and not args.eval:
        parser.error('--checkpoint is only supported for --eval; no implicit resume')
    config = read_json(args.config)
    if args.static_check or args.preflight:
        checks = preflight(ROOT, config, args.model_dir, require_gpu=args.preflight)
        if args.output_root:
            args.output_root.mkdir(parents=True, exist_ok=False)
            write_json(args.output_root / 'preflight.json', checks)
        print(json.dumps(checks, indent=2))
        return 0 if checks['status'] == 'passed' else 2
    if not args.output_root or args.output_root.exists():
        parser.error('Provide a new --output-root; existing output is never overwritten')
    if args.cpu_smoke:
        load_data(ROOT, config)
        from src.training.grpo_v1.synthetic import cpu_smoke
        print(json.dumps(cpu_smoke(config, args.output_root), indent=2))
        return 0
    checks = preflight(ROOT, config, args.model_dir)
    if checks['status'] != 'passed':
        print(json.dumps(checks, indent=2))
        return 2
    if args.eval and not args.checkpoint:
        parser.error('--eval requires --checkpoint')
    args.output_root.mkdir(parents=True, exist_ok=False)
    try:
        if args.train:
            result = run_training(config, args.model_dir, args.output_root, checks)
        else:
            from src.training.grpo_v1.policy import load_policy
            from src.training.grpo_v1.evaluation import greedy_eval
            checkpoint_metadata = read_json(args.checkpoint / 'metadata.json')
            if checkpoint_metadata['config'] != config:
                raise ValueError('Checkpoint config does not match this smoke')
            _, diagnostic = load_data(ROOT, config)
            policy = load_policy(args.model_dir, config, checkpoint=args.checkpoint)
            result = greedy_eval(diagnostic, policy, config, args.output_root / 'diagnostic')
        print(json.dumps(result, indent=2))
    except Exception as exc:
        write_json(args.output_root / 'status.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc)))
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
