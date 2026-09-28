"""Offline ground-truth diagnostics. Never passed to policy or used as a reward."""
from collections import Counter, defaultdict
import json
from statistics import pvariance
from . import normalize, normalized_equal


def evaluate_policy(tasks, trajectories):
    if not trajectories: raise ValueError('No trajectories to evaluate')
    indexed = {t['task_id']: t for t in tasks}
    rows, grouped = [], defaultdict(list)
    action_counts, first_actions, tool_counts = Counter(), Counter(), Counter()
    proposed_actions, first_proposals = Counter(), Counter()
    repeated_calls, list_strings, parse_errors = [], [], Counter()
    for t in trajectories:
        task = indexed[t.task_id]
        complete = t.termination_reason == 'direct_answer' and not t.truncated
        counts = Counter(c['tool_name'] for c in t.tool_calls)
        tool_counts.update(counts)
        action_counts.update(t.tool_sequence)
        if t.tool_sequence: first_actions[t.tool_sequence[0]] += 1
        else: first_actions['no_executed_action'] += 1
        errors = sum(not p['success'] for p in t.parse_status)
        signatures = [json.dumps([c['tool_name'], c['arguments']], sort_keys=True) for c in t.tool_calls]
        if len(signatures) > len(set(signatures)):
            repeated_calls.append({'task_id': t.task_id, 'rollout_id': t.metadata['rollout_id'],
                                   'repeated_calls': len(signatures)-len(set(signatures))})
        if isinstance(t.final_answer, str) and t.final_answer.strip().startswith('['):
            list_strings.append({'task_id': t.task_id, 'rollout_id': t.metadata['rollout_id']})
        for status in t.parse_status:
            if not status['success']: parse_errors[status['error']] += 1
            parsed = status.get('parsed_action')
            proposal = parsed.get('action') if isinstance(parsed, dict) else None
            if not isinstance(proposal, str): proposal = 'unparsed'
            proposed_actions[proposal] += 1
            if status['step'] == 1 and status['retry_count'] == 0: first_proposals[proposal] += 1
        rows.append({
            'task_id': t.task_id, 'task_type': task['task_type'], 'rollout_id': t.metadata['rollout_id'],
            'correctness': bool(complete and normalized_equal(t.final_answer, task['ground_truth'])),
            'total_cost': t.total_cost, 'number_of_tool_calls': len(t.tool_calls),
            'deep_search_used': bool(counts['deep_search']),
            'direct_answer_without_tool': complete and not t.tool_calls,
            'termination_reason': t.termination_reason, 'truncated': t.truncated,
            'invalid_action_attempts': errors, 'action_attempts': len(t.parse_status),
            'invalid_arguments': sum(p['invalid_arguments'] for p in t.parse_status),
            'repair_attempts': sum(p['retry_count'] > 0 for p in t.parse_status),
            'tool_execution_errors': sum('error' in c['observation'] for c in t.tool_calls),
            'tool_sequence': t.tool_sequence, 'final_answer': t.final_answer,
        })
        grouped[t.task_id].append(t)
    def aggregate(selected):
        n = len(selected)
        average = lambda key: sum(r[key] for r in selected)/n
        attempts = sum(r['action_attempts'] for r in selected)
        return {
            'num_trajectories': n, 'normalized_accuracy': average('correctness'),
            'average_cost': average('total_cost'), 'average_tool_calls': average('number_of_tool_calls'),
            'deep_search_usage_rate': average('deep_search_used'),
            'direct_answer_without_tool_rate': average('direct_answer_without_tool'),
            'invalid_action_rate': sum(r['invalid_action_attempts'] for r in selected)/attempts if attempts else 0,
            'episodes_with_invalid_action_rate': sum(r['invalid_action_attempts'] > 0 for r in selected)/n,
            'truncation_rate': average('truncated'),
            'tool_execution_errors': sum(r['tool_execution_errors'] for r in selected),
            'termination_counts': dict(Counter(r['termination_reason'] for r in selected)),
        }
    diversity = {}
    for task_id, group in grouped.items():
        answers = [json.dumps(normalize(t.final_answer), sort_keys=True, ensure_ascii=False) for t in group if t.termination_reason == 'direct_answer']
        diversity[task_id] = {
            'group_size': len(group),
            'unique_tool_sequences': len({tuple(t.tool_sequence) for t in group}),
            'unique_final_answers': len(set(answers)),
            'missing_final_answers': sum(t.termination_reason != 'direct_answer' for t in group),
            'cost_variance': pvariance(t.total_cost for t in group),
            'rollouts': [{'rollout_id':t.metadata['rollout_id'], 'tool_sequence':t.tool_sequence,
                          'final_answer':t.final_answer, 'cost':t.total_cost, 'termination_reason':t.termination_reason} for t in group],
        }
    def shares(counts):
        total = sum(counts.values())
        return {action: {'count': counts[action], 'fraction': counts[action]/total if total else 0}
                for action in ('lookup', 'search', 'deep_search', 'direct_answer')}
    return {'aggregate': aggregate(rows),
            'by_task_type': {kind: aggregate([r for r in rows if r['task_type'] == kind]) for kind in sorted({r['task_type'] for r in rows})},
            'action_usage': shares(action_counts), 'tool_call_usage': {k:v for k,v in shares(tool_counts).items() if k != 'direct_answer'},
            'first_action_counts': dict(first_actions),
            'proposed_action_counts': dict(proposed_actions), 'first_proposed_action_counts': dict(first_proposals),
            'diagnostics': {'episodes_with_identical_repeated_calls': repeated_calls,
                            'list_like_string_answers': list_strings, 'parse_error_counts': dict(parse_errors)},
            'diversity': diversity, 'task_level': rows}


def write_report(path, summary, trajectories):
    from pathlib import Path
    def block(value): return '```json\n' + json.dumps(value, indent=2, ensure_ascii=False) + '\n```\n'
    lines = ['# Base Policy Tool-Use Distribution\n',
             '真实 HuggingFace 推理诊断，无 AI Judge、reward 优化或训练。CPU smoke 只检验接口，不代表 40–80 条正式实验。\n',
             '## Run configuration\n', block(summary['run']),
             '## Aggregate\n', block(summary['aggregate']),
             '## Task type\n', '| Type | N | Accuracy | Average cost |\n|---|---:|---:|---:|']
    for kind, row in summary['by_task_type'].items():
        lines.append(f"| {kind} | {row['num_trajectories']} | {row['normalized_accuracy']:.3f} | {row['average_cost']:.3f} |")
    lines += ['\n## Tool / action usage\n', 'action_usage 包括终止 action；tool_call_usage 仅以已执行工具调用为分母。\n',
              'proposed_action_counts 包含可解析出 action 名称但参数验证失败的尝试；unparsed 表示不能可靠提取名称。不要将零次实际调用误解为没有使用工具的意图。\n',
              block({key: summary[key] for key in ('action_usage', 'tool_call_usage', 'first_action_counts', 'proposed_action_counts', 'first_proposed_action_counts')}),
              '## Group differences\n', block(summary['diversity']),
              '## Invalid / truncated\n',
              'invalid_action_rate = 解析或 schema 验证失败次数 / 所有模型 action 输出次数（包括 repair）。达到最终解析失败时以 invalid_action 结束，不自动编造答案；truncated 专指预算/上下文/推理中断。工具执行错误另外计数。\n',
              block([r for r in summary['task_level'] if r['invalid_action_attempts'] or r['truncated'] or r['tool_execution_errors']]),
              '## Observed interface signals\n',
              '以下仅标记重复的相同调用和列表样式字符串，不自动修正模型答案或影响 policy。\n',
              block(summary['diagnostics']),
              '## Interface limitations\n',
              '- search 的 top-k 无分页；列表题可能需要 deep_search。工具能力不对称会影响初始偏好。\n',
              '- deep_search 返回数据库查询结果和较多扫描证据；长 observation 可能耗尽上下文。当前显式终止，不静默裁掉历史。\n',
              '- keyword search 是子串查询；自然语言问题不能直接当作 deep_search query。\n',
              '- schema 合法不等于存在对应实体；此类错误由工具报告，并按原环境规则计费。\n',
              '- 最终答案只接受标量或标量数组；正确率不理解自然语言解释。\n',
              '- 独立随机种子不保证行为不同，低 diversity 本身是诊断结果。\n',
              '## Five complete trajectory examples\n']
    # Deterministically spread examples across questions before taking additional group members.
    examples, seen = [], set()
    for t in trajectories:
        if t.task_id not in seen:
            examples.append(t); seen.add(t.task_id)
    selected_ids = {id(t) for t in examples[:5]}
    examples = examples[:5] + [t for t in trajectories if id(t) not in selected_ids][:max(0, 5-len(examples))]
    for i, t in enumerate(examples, 1):
        lines += [f'### Example {i}: {t.task_id} / rollout {t.metadata["rollout_id"]}\n', block(t.to_dict())]
    if len(trajectories) < 5: lines.append('本次不足 5 条真实轨迹，未用模拟样本补齐。\n')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
