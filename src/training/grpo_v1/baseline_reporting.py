"""Offline baseline accounting and descriptive reports; never shape rewards."""
from collections import Counter
import json
from pathlib import Path

from src.evaluation.model_scale import annotate


def group_metrics(group, update, attempted, elapsed_seconds, memory):
    rewards = [row['reward'] for row in group]
    if len(group) != 4 or len({row['task_id'] for row in group}) != 1:
        raise ValueError('Expected one G=4 task group')
    mixed = min(rewards) != max(rewards)
    repeated = [annotate(row)['repeated_identical_action_count'] for row in group]
    mean = lambda values: sum(values) / len(values)
    return dict(task_id=group[0]['task_id'], difficulty=group[0]['difficulty'],
        seed=group[0]['seed'], attempted_groups=attempted, rewards=rewards,
        **update, optimizer_step_performed=update['update_applied'],
        mean_reward=mean(rewards), success_rate=mean([row['success'] for row in group]),
        mixed_reward_group=mixed, all_success_group=all(rewards), all_failure_group=not any(rewards),
        trajectory_lengths=[row['trajectory_length'] for row in group],
        avg_trajectory_length=mean([row['trajectory_length'] for row in group]),
        invalid_action=[row['termination_reason'] == 'invalid_action' for row in group],
        invalid_action_rate=sum(row['termination_reason'] == 'invalid_action' for row in group) / 4,
        max_tool_calls=[row['termination_reason'] == 'max_tool_calls' for row in group],
        max_tool_call_rate=sum(row['termination_reason'] == 'max_tool_calls' for row in group) / 4,
        repeated_actions=repeated, repeated_action_count=sum(repeated),
        tool_costs=[row['tool_cost'] for row in group], avg_tool_cost=mean([row['tool_cost'] for row in group]),
        violations=[row['rule_violations'] for row in group],
        rule_violation_count=sum(len(row['rule_violations']) for row in group),
        generated_token_counts=[row['generated_token_count'] for row in group],
        generated_token_count=sum(row['generated_token_count'] for row in group),
        elapsed_seconds=elapsed_seconds, **memory)


def training_summary(groups):
    count = len(groups)
    mixed = sum(row['mixed_reward_group'] for row in groups)
    all_success = sum(row['all_success_group'] for row in groups)
    all_failure = sum(row['all_failure_group'] for row in groups)
    rewards = [reward for row in groups for reward in row['rewards']]
    levels = {}
    for level in ('L1', 'L2', 'L3'):
        selected = [reward for row in groups if row['difficulty'] == level for reward in row['rewards']]
        levels[level] = dict(trajectories=len(selected), successes=sum(selected),
                             success_rate=sum(selected) / len(selected) if selected else None)
    ratio_tokens = sum(row.get('ratio_token_count', 0) for row in groups)
    measured = [row for row in groups if row.get('ratio_token_count', 0)]
    weighted = lambda key: (sum(row[key] * row['ratio_token_count'] for row in measured) / ratio_tokens
                            if ratio_tokens else None)
    return dict(attempted_groups=count, mixed_group_count=mixed,
        mixed_reward_group_rate=mixed / count if count else None,
        all_success_group_count=all_success, all_success_group_rate=all_success / count if count else None,
        all_failure_group_count=all_failure, all_failure_group_rate=all_failure / count if count else None,
        optimizer_steps=sum(row['optimizer_step_performed'] for row in groups),
        zero_variance_groups=sum(row.get('zero_variance_group', not row['mixed_reward_group']) for row in groups),
        mean_reward_variance=(sum(row.get('reward_variance', 0.0) for row in groups) / count if count else None),
        ratio_mean=weighted('ratio_mean'),
        clip_fraction=weighted('clip_fraction'), approx_kl=weighted('approx_kl'),
        ratio_token_count=ratio_tokens,
        trajectories=len(rewards), cumulative_success_count=sum(rewards),
        cumulative_task_success=sum(rewards) / len(rewards) if rewards else None,
        success_by_difficulty=levels, reward_distribution=dict(Counter(str(int(r)) for r in rewards)),
        group_success_count_distribution=dict(Counter(str(int(sum(row['rewards']))) for row in groups)),
        generated_token_count=sum(row['generated_token_count'] for row in groups),
        peak_vram_bytes=max((row['gpu_peak_memory'] or 0 for row in groups), default=0) or None)


def render_report(directory):
    directory = Path(directory)
    def read(name):
        path = directory / name
        return json.loads(path.read_text()) if path.exists() else None
    dry = read('dry_run.json')
    status = read('status.json') or dict(status='not_run')
    summary = read('training_summary.json') or training_summary([])
    before = read('base_dev/summary.json')
    after = read('eval_group_240/dev/summary.json')
    diagnostic = read('eval_group_240/diagnostic/summary.json')
    archived_diag = read('base_nf4_diagnostic.json')
    def fmt(value):
        if value is None:
            return '—'
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    def value(metrics, key):
        if metrics is None:
            return None
        if key.startswith('level_'):
            return metrics.get('success_by_difficulty', {}).get(key[6:])
        return metrics.get(key)
    fields = [('overall success', 'overall_success'), ('L1 success', 'level_L1'),
        ('L2 success', 'level_L2'), ('L3 success', 'level_L3'), ('invalid_action rate', 'invalid_action_rate'),
        ('max-tool-call rate', 'max_tool_call_rate'), ('repeated actions (total)', 'repeated_identical_action_count'),
        ('avg cost', 'average_tool_cost'), ('avg cost successful', 'average_tool_cost_successful'),
        ('success & zero violation', 'success_zero_violation'), ('rule violation count', 'rule_violation_count'),
        ('deep_search usage', 'deep_search_usage')]
    lines = ['# GRPO v1 task-only baseline results', '', f'状态：{status["status"]}。', '',
        '模型：Qwen2.5-7B-Instruct，NF4/bf16 + LoRA。reward仅为原evaluator task_success。',
        '单次固定240题训练，G=4；dev为独立全部400题，diagnostic为固定15题，仅最终checkpoint评估。', '']
    if dry:
        lines += [f'选题hash：`{dry["selection_sha256"]}`。',
                  f'计划：{dry["attempted_groups"]} groups，{dry["training_rollouts"]} sampled trajectories，最多{dry["maximum_optimizer_steps"]} optimizer steps。', '']
    lines += ['## Dev：同一NF4 base vs 最终task-only GRPO', '', '| 指标 | Base NF4 | GRPO task-only |', '|---|---:|---:|']
    lines += [f'| {label} | {fmt(value(before, key))} | {fmt(value(after, key))} |' for label, key in fields]
    lines += ['', '## 最终diagnostic', '', '| 指标 | 既有Base NF4（若提供） | 最终GRPO |', '|---|---:|---:|']
    lines += [f'| {label} | {fmt(value(archived_diag, key))} | {fmt(value(diagnostic, key))} |' for label, key in fields]
    lines += ['', '没有提供可验证的既有NF4 diagnostic记录时，Base列保持缺失；不拿bf16基线替代，不在训练中运行diagnostic。', '',
              '## 训练过程', '', f'- attempted groups：{summary["attempted_groups"]}',
              f'- mixed groups / rate：{summary["mixed_group_count"]} / {fmt(summary["mixed_reward_group_rate"])}',
              f'- all-success groups / rate：{summary["all_success_group_count"]} / {fmt(summary["all_success_group_rate"])}',
              f'- all-failure groups / rate：{summary["all_failure_group_count"]} / {fmt(summary["all_failure_group_rate"])}',
              f'- optimizer steps：{summary["optimizer_steps"]}',
              f'- reward distribution：{json.dumps(summary["reward_distribution"])}',
              f'- cumulative task success：{fmt(summary["cumulative_task_success"])}',
              f'- success by difficulty：{json.dumps(summary["success_by_difficulty"])}',
              f'- peak VRAM bytes：{fmt(status.get("gpu_peak_memory", summary["peak_vram_bytes"]))}',
              f'- wall-clock seconds（含加载/eval/checkpoint）：{fmt(status.get("wall_clock_seconds"))}', '',
              '## Dev curve', '', '| 已处理groups | optimizer steps | dev success | L1 | L2 | L3 |', '|---:|---:|---:|---:|---:|---:|']
    for attempted in (0, 60, 120, 180, 240):
        evaluation = before if attempted == 0 else read(f'eval_group_{attempted:03}/dev/summary.json')
        progress = read(f'checkpoint_group_{attempted:03}.json') or {}
        lines.append(f'| {attempted} | {fmt(progress.get("optimizer_steps"))} | {fmt(value(evaluation,"overall_success"))} | '
                     f'{fmt(value(evaluation,"level_L1"))} | {fmt(value(evaluation,"level_L2"))} | {fmt(value(evaluation,"level_L3"))} |')
    if status['status'] == 'completed' and before and after:
        delta = after['overall_success'] - before['overall_success']
        direction = '提高' if delta > 0 else '下降' if delta < 0 else '不变'
        lines += ['', f'本轮固定dev集success{direction}，变化{delta * 100:+.2f}个百分点。单次运行的描述性结果，不据此修改环境、任务或reward。']
    else:
        lines += ['', '正式运行尚未完整结束，不能回答task-only GRPO是否改善policy；缺失结果不是0分。']
    path = directory / 'GRPO_V1_BASELINE_RESULTS.md'
    path.write_text('\n'.join(lines) + '\n')
    return path
