"""Offline diagnostics only: never modify observations, actions, or task success."""
from collections import Counter
from copy import deepcopy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_tasks(tasks, task_ids=None, count=40):
    if not 1 <= count <= len(tasks):
        raise ValueError('task-count out of range')
    if task_ids is None:
        return tasks[:count]
    if not isinstance(task_ids, list) or not task_ids or any(not isinstance(t, str) for t in task_ids):
        raise ValueError('task IDs must be a nonempty string list')
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('Duplicate task IDs')
    indexed = {t['task_id']: t for t in tasks}
    if any(t not in indexed for t in task_ids):
        raise ValueError('Unknown task ID')
    return [indexed[t] for t in task_ids][:count]


def annotate(row):
    result = deepcopy(row)
    calls = row['trajectory']['tool_calls']
    keys = [json.dumps([c['tool_name'], c['arguments']], sort_keys=True) for c in calls]
    result.update(
        termination_reason=row['trajectory']['termination_reason'],
        deep_search_calls=sum(c['tool_name'] == 'deep_search' for c in calls),
        repeated_identical_action_count=sum(a == b for a, b in zip(keys, keys[1:])),
        answer_ready_but_continued=None,
    )
    return result


def diagnostics(rows):
    rows = [annotate(r) for r in rows]
    if not rows:
        raise ValueError('No diagnostic results')
    repeated = sum(r['repeated_identical_action_count'] for r in rows)
    return dict(
        termination_counts=dict(Counter(r['termination_reason'] for r in rows)),
        max_tool_call_rate=sum(r['termination_reason'] == 'max_tool_calls' for r in rows) / len(rows),
        repeated_identical_action_count=repeated,
        average_repeated_identical_action_count=repeated / len(rows),
        deep_search_calls=sum(r['deep_search_calls'] for r in rows),
        answer_ready_but_continued=None,
        answer_ready_method='not_inferred; consecutive identical executed tool+arguments calls used as the permitted proxy',
    )
