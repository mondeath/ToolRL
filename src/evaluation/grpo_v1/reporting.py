"""Independent paired metrics; unavailable token counts remain null."""
import json
from collections import Counter
from src.env_v1.evaluation import summarize


def metrics(rows):
    result = summarize(rows)
    successes = sum(r['task_success'] for r in rows)
    counts = [r.get('generated_tokens') for r in rows]
    repeats = []
    for r in rows:
        calls = [json.dumps([c['tool_name'], c['arguments']], sort_keys=True) for c in r['trajectory']['tool_calls']]
        repeats.append(sum(n - 1 for n in Counter(calls).values()))
    result.update(task_success_rate=successes / len(rows),
        cost_per_successful_task=sum(r['tool_cost'] for r in rows) / successes if successes else None,
        average_cost_successful=sum(r['tool_cost'] for r in rows if r['task_success']) / successes if successes else None,
        generated_tokens=sum(counts) if all(n is not None for n in counts) else None,
        average_generated_tokens=sum(counts) / len(counts) if all(n is not None for n in counts) else None,
        token_count_coverage=sum(n is not None for n in counts) / len(counts),
        error_recovery_rate=result['recovery_success'], repeated_actions=sum(repeats),
        average_repeated_actions=sum(repeats) / len(rows),
        invalid_action_rate=sum(r['trajectory']['termination_reason'] == 'invalid_action' for r in rows) / len(rows),
        max_tool_call_rate=sum(r['trajectory']['termination_reason'] == 'max_tool_calls' for r in rows) / len(rows))
    return result


def comparison(before, after):
    def index(rows):
        out = {r['task_id']: r for r in rows}
        if len(out) != len(rows): raise ValueError('Duplicate task ID')
        return out
    a, b = index(before), index(after)
    if a.keys() != b.keys(): raise ValueError('Unpaired task lists')
    pairs = []
    for tid, x in a.items():
        y = b[tid]
        if x['difficulty'] != y['difficulty']: raise ValueError('Difficulty mismatch')
        state = ('both_success' if x['task_success'] else 'improved') if y['task_success'] else ('regressed' if x['task_success'] else 'both_failure')
        pairs.append(dict(task_id=tid, difficulty=x['difficulty'], outcome=state,
            base_success=x['task_success'], trained_success=y['task_success'],
            tool_calls_delta=y['tool_calls']-x['tool_calls'], tool_cost_delta=y['tool_cost']-x['tool_cost'],
            generated_tokens_delta=y['generated_tokens']-x['generated_tokens'] if x.get('generated_tokens') is not None and y.get('generated_tokens') is not None else None,
            base_failure_events=len(x['failure_events']), trained_failure_events=len(y['failure_events'])))
    exposed = [p for p in pairs if p['base_failure_events'] and p['trained_failure_events']]
    return dict(base=metrics(before), trained=metrics(after),
        by_difficulty={level: dict(base=metrics([r for r in before if r['difficulty']==level]),
            trained=metrics([r for r in after if r['difficulty']==level])) for level in sorted({r['difficulty'] for r in before})},
        paired_counts=dict(Counter(p['outcome'] for p in pairs)), paired_tasks=pairs,
        both_exposed_count=len(exposed),
        both_exposed_success_rate={key: sum(p[key+'_success'] for p in exposed)/len(exposed) if exposed else None for key in ('base','trained')},
        failure_cases=[dict(task_id=tid, base=a[tid], trained=b[tid]) for tid in a if not a[tid]['task_success'] or not b[tid]['task_success']])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def save_pair(directory, before, after):
    directory.mkdir(parents=True, exist_ok=False)
    for name, rows in [('base',before),('trained',after)]:
        (directory / (name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))
    result = comparison(before,after)
    write_json(directory/'comparison.json', result)
    keys = ['task_success_rate','rule_violation_rate','average_tool_calls','average_tool_cost','cost_per_successful_task','generated_tokens','error_recovery_rate']
    text = '# Independent paired evaluation\n\n| Metric | Base NF4 | GRPO NF4 |\n|---|---:|---:|\n'
    text += ''.join(f'| {k} | {result["base"][k]} | {result["trained"][k]} |\n' for k in keys)
    text += '\nPer difficulty: '+json.dumps({k: {m:v[m]['task_success_rate'] for m in ('base','trained')} for k,v in result['by_difficulty'].items()})+'\n'
    text += '\nPaired outcomes: '+json.dumps(result['paired_counts'])+'\n\nFull trajectories and failure cases: base.jsonl, trained.jsonl, comparison.json.\n'
    (directory/'REPORT.md').write_text(text)
    return result
