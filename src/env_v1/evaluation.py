"""Final database truth and separate diagnostic metrics; no reward shaping."""
from copy import deepcopy
from src.evaluation import normalized_equal
from src.tools import QueryEngine
from . import matching_students, eligibility


def evaluate(task, env):
    t, before, after = env.trajectory, env.initial_kb, env.engine.kb
    violations = deepcopy(env.violations)
    if any(after.get(table) != rows for table, rows in before.items() if table != 'applications') or set(after) != set(before):
        violations.append(dict(type='illegal_state_mutation', reason='non-application tables changed'))
    apps = after.get('applications', {})
    if any(apps.get(aid) != row for aid, row in before['applications'].items()):
        violations.append(dict(type='illegal_state_mutation', reason='existing application changed'))
    seen = set()
    for aid, row in apps.items():
        pair = (row.get('student_id'), row.get('scholarship_id'))
        if pair in seen:
            violations.append(dict(type='illegal_state_mutation', reason='duplicate application state'))
        seen.add(pair)
        if aid not in before['applications']:
            try:
                valid = eligibility(before, *pair)['eligible'] and row == dict(id=f'{pair[0]}:{pair[1]}', student_id=pair[0], scholarship_id=pair[1], status='submitted') and aid == row['id']
            except (KeyError, TypeError): valid = False
            if not valid: violations.append(dict(type='illegal_state_mutation', reason='invalid new application'))
    spec = task['evaluation']
    assertion_results = []
    for assertion in task['final_assertions']:
        if assertion['type'] != 'applications_equal': raise ValueError('Unknown final assertion')
        assertion_results.append(apps == assertion['expected'])
    for assertion in task['forbidden_assertions']:
        if assertion['type'] == 'non_application_mutation':
            assertion_results.append(all(after.get(table) == rows for table, rows in before.items() if table != 'applications') and set(before) == set(after))
        elif assertion['type'] == 'existing_application_mutation':
            assertion_results.append(all(apps.get(aid) == row for aid, row in before['applications'].items()))
        else: raise ValueError('Unknown forbidden assertion')
    if spec['kind'] == 'state':
        if not task['final_assertions']: raise ValueError('State task requires explicit final assertions')
        correct = all(assertion_results)
    else:
        engine = QueryEngine(before)
        if spec['kind'] == 'lookup': expected = engine.lookup(spec['entity'], spec['field'])['value']
        elif spec['kind'] == 'relational': expected = matching_students(before, spec['query'])
        else:
            result = engine.execute(spec['query'])
            expected = result['matched_count'] if spec['kind'] == 'count' else result['answer']
        correct = normalized_equal(t.final_answer, expected) and all(assertion_results)
        if apps != before['applications']:
            violations.append(dict(type='illegal_state_mutation', reason='read-only task mutated applications'))
    completed = bool(t.steps and t.steps[-1]['action'] == 'direct_answer' and not t.truncated)
    success = bool(correct and completed and not any(v['type'] == 'illegal_state_mutation' for v in violations))
    return dict(task_id=task['task_id'], difficulty=task['difficulty'], success=success, task_success=success,
                rule_violation_count=len(violations), rule_violations=violations,
                tool_cost=sum(c['cost'] for c in t.tool_calls), trajectory_length=len(t.steps),
                tool_calls=len(t.tool_calls), expensive_tool_calls=sum(c['tool_name'] == 'deep_search' for c in t.tool_calls),
                recovery_success=success if env.failures else None,
                failure_type=sorted({f['type'] for f in env.failures}), failure_events=deepcopy(env.failures),
                completed=completed, final_state_correct=bool(correct), assertion_results=assertion_results,
                trajectory=t.to_dict(), final_applications=deepcopy(apps))


def summarize(rows):
    if not rows: raise ValueError('No results')
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    injected = [r for r in rows if r['recovery_success'] is not None]
    return dict(task_count=len(rows), overall_success=mean('success'),
                success_by_difficulty={level: sum(r['success'] for r in rows if r['difficulty'] == level) / sum(r['difficulty'] == level for r in rows) for level in sorted({r['difficulty'] for r in rows})},
                average_tool_cost=mean('tool_cost'), average_trajectory_length=mean('trajectory_length'),
                average_tool_calls=mean('tool_calls'), deep_search_usage=sum(r['expensive_tool_calls'] > 0 for r in rows) / len(rows),
                deep_search_calls=sum(r['expensive_tool_calls'] for r in rows),
                rule_violation_rate=sum(r['rule_violation_count'] > 0 for r in rows) / len(rows),
                rule_violation_count=sum(r['rule_violation_count'] for r in rows),
                recovery_success=sum(r['recovery_success'] for r in injected) / len(injected) if injected else None,
                injected_episodes=len(injected))
