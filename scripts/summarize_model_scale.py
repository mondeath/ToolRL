"""Read archived episodes only. Never load a model, execute tools, or re-evaluate success."""
from _common import ROOT
import argparse
from collections import Counter
import hashlib
import json
from src.evaluation.model_scale import annotate, diagnostics, digest

SOURCES = {
    '1.5B': ROOT/'results/model_scale/qwen2.5-1.5b',
    '3B': ROOT/'results/model_scale_cuda_bf16/qwen2.5-3b',
    '7B': ROOT/'results/model_scale_cuda_bf16/qwen2.5-7b',
}


def mean_cost(rows):
    return sum(r['tool_cost'] for r in rows)/len(rows) if rows else None


def statistics(rows):
    if not rows:
        raise ValueError('No archived episodes')
    for row in rows:
        if type(row['success']) is not bool or row['success'] != row['task_success']:
            raise ValueError('Inconsistent saved task success')
        calls = row['trajectory']['tool_calls']
        if row['tool_calls'] != len(calls) or row['tool_cost'] != sum(c['cost'] for c in calls):
            raise ValueError('Saved tool accounting mismatch')
        recalculated = annotate(row)
        for key in ('termination_reason', 'deep_search_calls', 'repeated_identical_action_count'):
            if row.get(key, recalculated[key]) != recalculated[key]:
                raise ValueError('Saved diagnostic mismatch: ' + key)
    success = [r for r in rows if r['success']]
    failure = [r for r in rows if not r['success']]
    terms = Counter(r['trajectory']['termination_reason'] for r in rows)
    attempts = [p for r in rows for p in r['trajectory']['parse_status']]
    return dict(
        episodes=len(rows), successes=len(success), failures=len(failure),
        overall=len(success)/len(rows),
        by_level={level: dict(episodes=len(rs), successes=sum(r['success'] for r in rs),
                             rate=sum(r['success'] for r in rs)/len(rs))
                  for level in sorted({r['difficulty'] for r in rows})
                  for rs in [[r for r in rows if r['difficulty'] == level]]},
        invalid_action_rate=terms['invalid_action']/len(rows),
        max_tool_call_rate=terms['max_tool_calls']/len(rows),
        repeated_actions=diagnostics(rows)['repeated_identical_action_count'],
        avg_cost=mean_cost(rows), avg_cost_success=mean_cost(success), avg_cost_failure=mean_cost(failure),
        by_termination={term: dict(episodes=len(rs), total_cost=sum(r['tool_cost'] for r in rs), avg_cost=mean_cost(rs))
                        for term in sorted(terms)
                        for rs in [[r for r in rows if r['trajectory']['termination_reason'] == term]]},
        parse_attempts=len(attempts), strict_parse_success=sum(p['strict_success'] for p in attempts),
        normalized_success=sum(p['normalized_success'] for p in attempts),
        failed_parse_attempts=sum(not p['success'] for p in attempts),
        zero_tool_episodes=sum(r['tool_calls'] == 0 for r in rows),
        deep_search_calls=sum(c['tool_name'] == 'deep_search' for r in rows for c in r['trajectory']['tool_calls']),
        clean_successes=sum(r['success'] and r['rule_violation_count'] == 0 for r in rows),
        rule_violations=dict(Counter(v['type'] for r in rows for v in r['rule_violations'])),
    )


def load_results():
    contract = json.loads((ROOT/'configs/env_v1_model_scale.json').read_text())
    ids = json.loads((ROOT/'configs/env_v1_diagnostic_tasks.json').read_text())['task_ids']
    full_tasks = json.loads((ROOT/'results/env_v1_baseline.tasks.json').read_text())
    tasks_by_id = {t['task_id']: t for t in full_tasks}
    selected = [tasks_by_id[task_id] for task_id in ids]
    baseline_path = ROOT/'results/env_v1_baseline.jsonl'
    full_rows = {r['task_id']: r for r in map(json.loads, baseline_path.read_text().splitlines())}
    baseline_prompt = json.loads((ROOT/'results/env_v1_baseline.prompt.json').read_text())
    results = {}
    for label, directory in SOURCES.items():
        source = directory/'results.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        summary = json.loads((directory/'summary.json').read_text())
        run = summary['run']
        if [r['task_id'] for r in rows] != ids or run['selected_task_ids'] != ids:
            raise ValueError('Diagnostic task IDs/order differ')
        if any(r['difficulty'] != tasks_by_id[r['task_id']]['difficulty'] for r in rows):
            raise ValueError('Difficulty labels differ')
        frozen = contract['frozen']
        if run['kb_sha256'] != frozen['kb_sha256'] or run['prompt_sha256'] != frozen['prompt_sha256'] or run['tasks_sha256'] != digest(selected):
            raise ValueError('Archived KB/prompt/task hashes differ')
        if run['failure_rate'] != 0 or run['rollout_config'] != frozen['rollout'] or any(run['model_config'][k] != v for k,v in frozen['policy'].items()):
            raise ValueError('Frozen generation/rollout settings differ')
        canonical = run.get('model_identity', {}).get('canonical_model_name') or run['model_config']['model_name']
        if canonical != f'Qwen/Qwen2.5-{label}-Instruct' or run['model_config']['revision'] != contract['models'][canonical]['revision']:
            raise ValueError('Model identity/revision differs')
        if run['model_config']['dtype'] != run['runtime']['dtype']:
            raise ValueError('Requested/actual dtype mismatch')
        if label == '1.5B':
            if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != summary['provenance']['source_sha256']:
                raise ValueError('Original baseline source changed')
            if any(any(r.get(k) != v for k,v in full_rows[r['task_id']].items()) for r in rows):
                raise ValueError('1.5B diagnostic differs from original saved episodes')
        else:
            if json.loads((directory/'results.tasks.json').read_text()) != selected or digest(json.loads((directory/'results.kb.json').read_text())) != run['kb_sha256']:
                raise ValueError('Saved input snapshots differ')
            if json.loads((directory/'results.prompt.json').read_text()) != baseline_prompt:
                raise ValueError('Actual prompt message snapshots differ')
        stats = statistics(rows)
        checks = {'task_count': stats['episodes'], 'overall_success': stats['overall'],
                  'average_tool_cost': stats['avg_cost'],
                  'success_by_difficulty': {k:v['rate'] for k,v in stats['by_level'].items()}}
        if any(summary[k] != v for k,v in checks.items()) or summary['diagnostics'] != diagnostics(rows):
            raise ValueError('Saved summary differs from JSONL recomputation')
        results[label] = dict(**stats, runtime=run['runtime'], revision=run['model_config']['revision'],
                              source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return results


def comparison_table(results):
    lines = ['| Model | Overall | L1 | L2 | L3 | Invalid action rate | Max-tool-call rate | Repeated actions | Avg cost | Avg cost (success) | Avg cost (failure) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    number = lambda value: 'N/A' if value is None else f'{value:.3f}'
    for model, s in results.items():
        lines.append(f'| Qwen2.5-{model}-Instruct | {s["overall"]:.1%} | '+
                     ' | '.join(f'{s["by_level"][level]["rate"]:.1%}' for level in ('L1','L2','L3'))+
                     f' | {s["invalid_action_rate"]:.1%} | {s["max_tool_call_rate"]:.1%} | {s["repeated_actions"]} | '+
                     ' | '.join(number(s[key]) for key in ('avg_cost','avg_cost_success','avg_cost_failure'))+' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print machine-readable statistics instead of the comparison table')
    args = parser.parse_args()
    results = load_results()
    print(json.dumps(results, indent=2, ensure_ascii=False) if args.json else comparison_table(results))


if __name__ == '__main__':
    main()
