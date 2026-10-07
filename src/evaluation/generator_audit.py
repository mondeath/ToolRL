"""Offline feasibility/data audit. Reference actions are never policy training data."""
from collections import Counter
from copy import deepcopy

from src.env_v1 import Environment, eligibility
from src.env_v1.tasks import initial_state
from src.env_v1.evaluation import evaluate
from src.evaluation.model_scale import digest
from src.tasks import public_task


def reference_path(kb, task):
    """Execute the existing 1/3/5-call reference strategies via actual tools."""
    env = Environment(initial_state(kb, task), failure_rate=0)
    env.reset(public_task(task))
    spec = task['evaluation']
    if task['difficulty'] == 'L1':
        if spec['kind'] == 'lookup':
            answer = env.step('lookup', entity=spec['entity'])['record'][spec['field']]
        elif spec['kind'] == 'count':
            answer = env.step('aggregate', query=spec['query'], operation='count')['answer']
        else:
            answer = env.step('search', query=spec['query'])['answer']
    elif spec['kind'] == 'state' or 'scholarship_id' in spec.get('query', {}):
        sc = spec.get('scholarship_id', spec.get('query', {}).get('scholarship_id'))
        scholarship = env.step('lookup', entity=sc)['record']
        candidates = env.step('search', query=dict(table='enrollments', where=[
            dict(field='course_id', value=scholarship['required_course'])], select='student_id'))['answer']
        checks = env.step('verify', student_ids=candidates, scholarship_id=sc)['results']
        answer = [c['student_id'] for c in checks if c['eligible']]
        if spec['kind'] == 'state':
            pending = [c['student_id'] for c in checks if c['eligible'] and not c['already_applied']]
            env.step('submit', student_ids=pending, scholarship_id=sc)
            env.step('verify', student_ids=answer, scholarship_id=sc)
    else:
        query = spec['query']
        students = env.step('search', query=dict(table='students', where=[
            dict(field='year', value=query['year'])]))['answer']
        completed = env.step('search', query=dict(table='enrollments', where=[
            dict(field='course_id', value=query['course_id']),
            dict(field='status', value='completed')], select='student_id'))['answer']
        high_gpa = env.step('search', query=dict(table='students', where=[
            dict(field='gpa', op='gte', value=query['min_gpa'])]))['answer']
        answer = sorted(set(students) & set(completed) & set(high_gpa))
    env.step('direct_answer', final_answer=answer)
    return evaluate(task, env)


def non_id_signature(kb):
    """A differing scalar multiset proves these KBs are not ID permutations.

    Ignore all names/IDs/references. This is a sufficient, not necessary, test
    for non-isomorphism; don't claim a general-purpose graph canonicalizer.
    """
    fields = {'students': ('gpa', 'year', 'department'),
              'courses': ('credits', 'department'),
              'enrollments': ('status', 'grade'),
              'scholarships': ('min_gpa', 'eligible_year')}
    return {table: sorted([list(row[field] for field in names) for row in kb[table].values()],
                          key=lambda values: repr(values)) for table, names in fields.items()}


def instance_fingerprint(kb, task):
    # Namespace/seed excluded. Include the actual initial state, not unrelated
    # metadata; tasks on different worlds are different instances even when an
    # individual answer or template coincides.
    spec = deepcopy(task)
    spec.pop('task_id')
    return digest(dict(initial_state=initial_state(kb, task), task=spec))


def audit_instance(kb, tasks):
    counts = Counter(task['difficulty'] for task in tasks)
    if counts != {'L1': 10, 'L2': 15, 'L3': 15}:
        raise ValueError(f'Unexpected task structure: {counts}')
    records = []
    for task in tasks:
        if task['difficulty'] == 'L3':
            state = initial_state(kb, task)
            checks = [eligibility(state, sid, task['evaluation']['scholarship_id']) for sid in state['students']]
            if not (sum(c['eligible'] for c in checks) == 3
                    and sum(c['eligible'] and c['already_applied'] for c in checks) == 1
                    and sum(c['eligible'] and not c['already_applied'] for c in checks) == 2
                    and any(not c['eligible'] for c in checks)):
                raise ValueError('L3 eligibility/application structure changed')
        row = reference_path(kb, task)
        expected_calls = {'L1': 1, 'L2': 3, 'L3': 5}[task['difficulty']]
        if not row['success'] or row['rule_violation_count'] or row['tool_calls'] != expected_calls:
            raise ValueError(f'Reference path failed: {task["task_id"]}')
        query = task['evaluation'].get('query', {})
        if task['difficulty'] == 'L2' and 'course_id' in query:
            expected_size = {3.5: 3, 3.65: 2}[query['min_gpa']]
            if len(row['trajectory']['final_answer']) != expected_size:
                raise ValueError('L2 threshold selectivity changed')
        outcome = row['final_applications'] if task['difficulty'] == 'L3' else row['trajectory']['final_answer']
        records.append(dict(task_id=task['task_id'], difficulty=task['difficulty'],
            instance_sha256=instance_fingerprint(kb, task), outcome_sha256=digest(outcome),
            reference_success=row['success'], reference_tool_calls=row['tool_calls']))
    return dict(kb_sha256=digest(kb), tasks_sha256=digest(tasks),
                non_id_signature_sha256=digest(non_id_signature(kb)), tasks=records)


def audit_splits(datasets):
    """Fail closed on split/seed reuse, instance duplication or ID-only novelty."""
    from src.env_v1.generation import SPLIT_SEEDS
    seen_seeds, seen_kbs, seen_signatures, seen_instances, seen_ids = set(), set(), set(), set(), set()
    output = []
    for split, seed, kb, tasks in datasets:
        if seed not in SPLIT_SEEDS.get(split, ()) or seed in seen_seeds:
            raise ValueError('Overlapping or unallocated split seed')
        seen_seeds.add(seed)
        audit = audit_instance(kb, tasks)
        for field, seen in [('kb_sha256', seen_kbs), ('non_id_signature_sha256', seen_signatures)]:
            if audit[field] in seen:
                raise ValueError(f'Duplicate world or no proof of non-ID novelty: {field}')
            seen.add(audit[field])
        for task in audit['tasks']:
            if task['instance_sha256'] in seen_instances or task['task_id'] in seen_ids:
                raise ValueError('Duplicate task instance or ID')
            seen_instances.add(task['instance_sha256'])
            seen_ids.add(task['task_id'])
        output.append(dict(split=split, seed=seed, **audit))
    return output
