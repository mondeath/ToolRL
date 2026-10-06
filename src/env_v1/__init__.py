"""Small deterministic relational environment; legacy environment remains unchanged."""
from copy import deepcopy
import hashlib
import random

from src.tools import QueryEngine
from src.environment import ToolEnvironment

FIELDS = {
    'students': ['id', 'name', 'year', 'gpa', 'department'],
    'courses': ['id', 'name', 'department', 'credits'],
    'enrollments': ['id', 'student_id', 'course_id', 'status', 'grade'],
    'scholarships': ['id', 'name', 'min_gpa', 'eligible_year', 'required_course'],
    'applications': ['id', 'student_id', 'scholarship_id', 'status'],
}


def generate_kb(seed=42):
    rng = random.Random(seed)
    kb = {table: {} for table in FIELDS}
    for i in range(12):
        cid = f'c{i:02}'
        kb['courses'][cid] = dict(id=cid, name=f'Course-{i:02}', department=['Logic', 'Robotics', 'Ecology'][i % 3], credits=3)
    # Five independent six-person cohorts. Every scholarship has eligible, ineligible,
    # already-applied and missing applicants; random filler courses preserve this invariant.
    for i in range(30):
        sid, group, member = f's{i:02}', i // 6, i % 6
        year = group % 4 + 1
        kb['students'][sid] = dict(id=sid, name=f'Student-{i:02}', year=year,
                                  gpa=[3.8, 3.7, 3.6, 3.1, 3.9, 3.4][member],
                                  department=['Logic', 'Robotics', 'Ecology'][group % 3])
        for j, cid in enumerate([f'c{group:02}'] + [f'c{x:02}' for x in rng.sample(range(5, 12), 2)]):
            eid = f'e{i:02}-{j}'
            kb['enrollments'][eid] = dict(id=eid, student_id=sid, course_id=cid,
                status='enrolled' if member == 4 and j == 0 else 'completed',
                grade=None if member == 4 and j == 0 else rng.choice(['A', 'B']))
    for group in range(5):
        sc = f'sc{group:02}'
        kb['scholarships'][sc] = dict(id=sc, name=f'Scholarship-{group:02}', min_gpa=3.5,
                                     eligible_year=group % 4 + 1, required_course=f'c{group:02}')
        sid = f's{group * 6:02}'
        aid = f'{sid}:{sc}'
        kb['applications'][aid] = dict(id=aid, student_id=sid, scholarship_id=sc, status='submitted')
    return kb


def eligibility(kb, student_id, scholarship_id):
    student, scholarship = kb['students'][student_id], kb['scholarships'][scholarship_id]
    checks = dict(gpa=student['gpa'] >= scholarship['min_gpa'],
                  year=student['year'] == scholarship['eligible_year'],
                  course=any(e['student_id'] == student_id and e['course_id'] == scholarship['required_course']
                             and e['status'] == 'completed' for e in kb['enrollments'].values()))
    return dict(student_id=student_id, scholarship_id=scholarship_id,
                eligible=all(checks.values()), checks=checks,
                already_applied=any(a['student_id'] == student_id and a['scholarship_id'] == scholarship_id
                                    for a in kb['applications'].values()))


def matching_students(kb, query):
    """Join on the SAME enrollment record; never independent any-match predicates."""
    if 'scholarship_id' in query:
        return [sid for sid in sorted(kb['students']) if eligibility(kb, sid, query['scholarship_id'])['eligible']]
    return [sid for sid, s in sorted(kb['students'].items())
            if s['year'] == query['year'] and s['gpa'] >= query['min_gpa']
            and any(e['student_id'] == sid and e['course_id'] == query['course_id'] and e['status'] == 'completed'
                    for e in kb['enrollments'].values())]


class Environment(ToolEnvironment):
    COSTS = dict(search=1, lookup=1, aggregate=1, submit=2, verify=1, deep_search=5)

    def __init__(self, kb, max_tool_calls=8, failure_rate=0.0, failure_type=None, seed=2026):
        if not 0 <= failure_rate <= 1 or failure_type not in (None, 'timeout', 'partial_result'):
            raise ValueError('Invalid failure profile')
        super().__init__(kb, top_k=150, max_tool_calls=max_tool_calls)
        self.initial_kb = deepcopy(kb)
        self.failure_rate, self.failure_type, self.seed = failure_rate, failure_type, seed
        from .protocol import ACTION_SCHEMA, parse_action
        self.action_schema, self.parse_action = ACTION_SCHEMA, parse_action

    def reset(self, task, metadata=None):
        observation = super().reset(task, metadata)
        self.engine = QueryEngine(deepcopy(self.initial_kb))
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{task["task_id"]}'.encode()).digest()[:8], 'big')
        self.rng = random.Random(seed)
        self.violations, self.failures = [], []
        return observation

    def violation(self, kind, **details):
        entry = dict(type=kind, **details)
        self.violations.append(entry)
        return dict(error=kind, **details)

    def _verify(self, student_ids, scholarship_id):
        if scholarship_id not in self.engine.kb['scholarships']:
            return self.violation('invalid_entity', entity=scholarship_id)
        results = []
        for sid in student_ids:
            if sid not in self.engine.kb['students']:
                results.append(self.violation('invalid_entity', entity=sid))
            else:
                results.append(eligibility(self.engine.kb, sid, scholarship_id))
        return dict(results=results)

    def _execute(self, action, arguments):
        kb = self.engine.kb
        if action == 'lookup':
            entity = arguments['entity']
            rows = [table[entity] for table in kb.values() if entity in table]
            if len(rows) != 1: return self.violation('invalid_entity', entity=entity)
            return dict(record=deepcopy(rows[0]))
        if action in ('search', 'aggregate'):
            query = arguments['query']
            result = self.engine.search(query, top_k=150)
            if action == 'search': return result
            values, op = result['answer'], arguments['operation']
            if op == 'count': answer = result['matched_count']
            elif op == 'list': answer = values
            else:
                # Preserve multiplicity: QueryEngine.answer intentionally deduplicates values.
                values = [v for row in result['results'] for v in row['values']]
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    raise ValueError('average/max require numeric values')
                answer = (sum(values) / len(values) if op == 'average' else max(values)) if values else None
            return dict(answer=answer)
        if action == 'deep_search':
            query = arguments['query']
            for key, table in [('scholarship_id', 'scholarships'), ('course_id', 'courses')]:
                if key in query and query[key] not in kb[table]:
                    return self.violation('invalid_entity', entity=query[key])
            return dict(answer=matching_students(kb, query))
        if action in ('verify', 'submit'):
            result = self._verify(**arguments)
            if action == 'verify' or 'error' in result: return result
            submitted = []
            for item in result['results']:
                if 'error' in item:
                    submitted.append(item)
                    continue
                sid, sc = item['student_id'], item['scholarship_id']
                # Recheck after each write to reject duplicates within one batch as well.
                current = eligibility(kb, sid, sc)
                if current['already_applied']:
                    submitted.append(self.violation('duplicate_submit', student_id=sid, scholarship_id=sc))
                elif not current['eligible']:
                    submitted.append(self.violation('ineligible_submit', student_id=sid, scholarship_id=sc))
                else:
                    aid = f'{sid}:{sc}'
                    kb['applications'][aid] = dict(id=aid, student_id=sid, scholarship_id=sc, status='submitted')
                    submitted.append(dict(student_id=sid, scholarship_id=sc, status='submitted'))
            return dict(results=submitted)
        raise ValueError('Unknown action')

    def step(self, action, **arguments):
        if action == 'direct_answer': return super().step(action, **arguments)
        if self.trajectory is None or self.done: raise RuntimeError('Reset required')
        if action not in self.COSTS: raise ValueError('Unknown action')
        t = self.trajectory
        if len(t.tool_calls) >= self.max_tool_calls: raise RuntimeError('Tool budget exhausted')
        # Validate direct callers too, before any mutation. Invalid calls still cost.
        import json
        parsed = self.parse_action(json.dumps(dict(action=action, arguments=arguments)))
        failure = None
        if not parsed.success:
            observation = dict(error=parsed.error, error_type='invalid_arguments')
            if action == 'submit': observation = self.violation('illegal_state_mutation', reason=parsed.error)
        else:
            if self.rng.random() < self.failure_rate:
                failure = self.failure_type or self.rng.choice(['timeout', 'partial_result'])
            if failure == 'timeout':
                observation = dict(error='Tool timed out; retry if needed', error_type='timeout')
            else:
                try:
                    observation = self._execute(action, arguments)
                except (KeyError, ValueError, TypeError) as exc:
                    observation = dict(error=str(exc), error_type=type(exc).__name__)
                if failure == 'partial_result' and 'error' not in observation:
                    # Remove redundant answer channels as well: no hidden complete-result leak.
                    if 'results' in observation:
                        observation['results'] = observation['results'][:-1]
                        observation.pop('answer', None)
                    elif isinstance(observation.get('answer'), list): observation['answer'] = observation['answer'][:-1]
                    elif 'record' in observation:
                        record = observation['record']
                        record.pop(next((f for f in ('gpa', 'required_course', 'status', 'credits') if f in record), 'name'), None)
                    else: observation.pop('answer', None)
                    observation['partial_result'] = True
        if failure: self.failures.append(dict(call_index=len(t.tool_calls), type=failure))
        call = dict(tool_name=action, arguments=deepcopy(arguments), observation=deepcopy(observation), cost=self.COSTS[action], failure_type=failure)
        t.tool_calls.append(call)
        t.observations.append(deepcopy(observation))
        t.steps.append(dict(action=action, tool_call_index=len(t.tool_calls)-1, cost=call['cost']))
        t.total_cost += call['cost']
        return deepcopy(observation)
