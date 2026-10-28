"""Frozen V2 hold-out worlds with new IDs, templates, and composition rules."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import random
import re
from pathlib import Path

from src.env_v1 import generate_kb as base_generate_kb, eligibility, matching_students
from src.env_v1.tasks import initial_state
from src.evaluation.generator_audit import instance_fingerprint
from src.evaluation.model_scale import digest
from src.tools import QueryEngine


GENERATOR_VERSION = 'grpo-v2-holdout-1'
HOLDOUT_SEEDS = tuple(range(3000, 3006))


def _rename_entities(kb):
    mappings = {
        'students': {key: 'h' + key for key in kb['students']},
        'courses': {key: 'h' + key for key in kb['courses']},
        'scholarships': {key: 'h' + key for key in kb['scholarships']},
        'enrollments': {key: 'h' + key for key in kb['enrollments']},
    }
    renamed = {table: {} for table in kb}
    for table in ('students', 'courses', 'scholarships', 'enrollments'):
        for old_id, row in kb[table].items():
            value = deepcopy(row)
            value['id'] = mappings[table][old_id]
            if table == 'scholarships':
                value['required_course'] = mappings['courses'][value['required_course']]
            elif table == 'enrollments':
                value['student_id'] = mappings['students'][value['student_id']]
                value['course_id'] = mappings['courses'][value['course_id']]
            renamed[table][value['id']] = value
    for row in kb['applications'].values():
        value = deepcopy(row)
        value['student_id'] = mappings['students'][value['student_id']]
        value['scholarship_id'] = mappings['scholarships'][value['scholarship_id']]
        value['id'] = value['student_id'] + ':' + value['scholarship_id']
        renamed['applications'][value['id']] = value
    return renamed


def generate_world(seed):
    if type(seed) is not int or seed not in HOLDOUT_SEEDS:
        raise ValueError('Seed is not allocated to frozen V2 hold-out')
    rng = random.Random(seed)
    kb = base_generate_kb(42)
    kb['enrollments'], kb['applications'] = {}, {}
    departments = ['Logic', 'Robotics', 'Ecology']
    core_courses = rng.sample(list(kb['courses']), 5)
    filler_courses = [cid for cid in kb['courses'] if cid not in core_courses]
    years = rng.sample([1, 2, 3, 4], 4) + [rng.randint(1, 4)]
    for course in kb['courses'].values():
        course.update(department=rng.choice(departments), credits=rng.choice([2, 3, 4]))
    for group in range(5):
        scholarship_id = f'sc{group:02}'
        course_id, year = core_courses[group], years[group]
        min_gpa = rng.choice([3.25, 3.4, 3.55, 3.7])
        kb['scholarships'][scholarship_id].update(
            min_gpa=min_gpa, eligible_year=year, required_course=course_id)
        members = [f's{group * 6 + index:02}' for index in range(6)]
        rng.shuffle(members)
        for role, student_id in enumerate(members):
            gpa = (round(rng.uniform(max(min_gpa, 3.55), 4.0), 2) if role < 3 else
                   round(rng.uniform(2.6, max(2.61, min_gpa - .05)), 2))
            student_year = year if role != 5 else (year % 4) + 1
            kb['students'][student_id].update(gpa=gpa, year=student_year,
                department=rng.choice(departments), name='Holdout-' + student_id)
            courses = [course_id] + rng.sample(filler_courses, 2)
            for index, enrolled_course in enumerate(courses):
                status = 'enrolled' if role == 4 and index == 0 else 'completed'
                enrollment_id = f'e{int(student_id[1:]):02}-{index}'
                kb['enrollments'][enrollment_id] = dict(id=enrollment_id, student_id=student_id,
                    course_id=enrolled_course, status=status,
                    grade=None if status == 'enrolled' else rng.choice(['A', 'B', 'C']))
        already = members[0]
        application_id = f'{already}:{scholarship_id}'
        kb['applications'][application_id] = dict(id=application_id, student_id=already,
            scholarship_id=scholarship_id, status='submitted')
    return _rename_entities(kb)


def _query_answer(kb, query, count=False):
    result = QueryEngine(kb).execute(query)
    return result['matched_count'] if count else result['answer']


def generate_tasks(kb, seed):
    tasks = []
    rng = random.Random(seed + 991)

    def add(level, template_id, question, spec, **extra):
        index = sum(task['difficulty'] == level for task in tasks)
        tasks.append(dict(task_id=f'test-{seed}-v2-{level}-{index:02}', difficulty=level,
            template_id=template_id, question=question, prompt=question, evaluation=spec,
            final_assertions=[], forbidden_assertions=[], failure_profile=None, **extra))

    students, courses = sorted(kb['students']), sorted(kb['courses'])
    scholarships = sorted(kb['scholarships'])
    sid, cid = rng.choice(students), rng.choice(courses)
    add('L1', 'student_department_lookup', f'Return the department of student {sid}.',
        dict(kind='lookup', entity=sid, field='department'))
    add('L1', 'course_credits_lookup', f'How many credits is course {cid} worth? Return a number.',
        dict(kind='lookup', entity=cid, field='credits'))
    department = rng.choice(['Logic', 'Robotics', 'Ecology'])
    threshold = rng.choice([3.2, 3.45, 3.6])
    query = dict(table='students', where=[dict(field='department', value=department),
        dict(field='gpa', op='gte', value=threshold)])
    add('L1', 'department_gpa_count',
        f'How many students are in {department} with GPA at least {threshold}? Return a number.',
        dict(kind='count', query=query))
    department, credits = rng.choice(['Logic', 'Robotics', 'Ecology']), rng.choice([2, 3, 4])
    query = dict(table='courses', where=[dict(field='department', value=department),
        dict(field='credits', value=credits)])
    add('L1', 'department_credit_course_list',
        f'List course IDs in {department} that carry {credits} credits.', dict(kind='query', query=query))

    department, year, threshold = rng.choice(['Logic', 'Robotics', 'Ecology']), rng.randint(1, 4), rng.choice([3.25, 3.5, 3.75])
    query = dict(table='students', where=[dict(field='department', value=department),
        dict(field='year', value=year), dict(field='gpa', op='gte', value=threshold)])
    add('L2', 'department_year_gpa_intersection',
        f'List student IDs in {department}, year {year}, with GPA at least {threshold}.',
        dict(kind='query', query=query))
    cid = rng.choice(courses)
    query = dict(table='enrollments', where=[dict(field='course_id', value=cid),
        dict(field='status', value='completed'), dict(field='grade', value='A')], select='student_id')
    add('L2', 'course_completion_grade_intersection',
        f'List student IDs who completed course {cid} with grade A.', dict(kind='query', query=query))
    year, threshold = rng.randint(1, 4), rng.choice([3.3, 3.5, 3.65])
    query = dict(table='scholarships', where=[dict(field='eligible_year', value=year),
        dict(field='min_gpa', op='gte', value=threshold)])
    add('L2', 'scholarship_year_threshold_intersection',
        f'List scholarship IDs for year {year} whose minimum GPA is at least {threshold}.',
        dict(kind='query', query=query))
    course_department, credits = rng.choice(['Logic', 'Robotics', 'Ecology']), rng.choice([2, 3, 4])
    query = dict(table='courses', where=[dict(field='department', value=course_department),
        dict(field='credits', op='gte', value=credits)])
    add('L2', 'course_department_minimum_credits',
        f'List course IDs in {course_department} with at least {credits} credits.',
        dict(kind='query', query=query))

    def add_state(template_id, batches, wording):
        initial = deepcopy(kb['applications'])
        target = deepcopy(initial)
        for scholarship_id, candidates in batches:
            for student_id in candidates:
                if eligibility(dict(kb, applications=target), student_id, scholarship_id)['eligible']:
                    application_id = f'{student_id}:{scholarship_id}'
                    if application_id not in target:
                        target[application_id] = dict(id=application_id, student_id=student_id,
                            scholarship_id=scholarship_id, status='submitted')
        add('L3', template_id, wording, dict(kind='state'), initial_applications=initial)
        tasks[-1]['final_assertions'] = [dict(type='applications_equal', expected=target)]
        tasks[-1]['forbidden_assertions'] = [dict(type='non_application_mutation'),
                                             dict(type='existing_application_mutation')]

    def candidates(scholarship_id, pending=1):
        eligible = [student_id for student_id in students if eligibility(kb, student_id, scholarship_id)['eligible']]
        already = [student_id for student_id in eligible if eligibility(kb, student_id, scholarship_id)['already_applied']]
        new = [student_id for student_id in eligible if student_id not in already]
        ineligible = [student_id for student_id in students if student_id not in eligible]
        return already[:1] + new[:pending] + ineligible[:1]

    sc0, sc1, sc2, sc3 = scholarships[:4]
    batch = candidates(sc0, 1)
    add_state('candidate_limited_submission', [(sc0, batch)],
        f'For scholarship {sc0}, consider only candidate IDs {batch}. Verify them and submit only candidates who are eligible and not already applied. Preserve every existing record, then finish with direct_answer.')
    left, right = candidates(sc1, 1), candidates(sc2, 1)
    add_state('two_scholarship_candidate_batches', [(sc1, left), (sc2, right)],
        f'Process two independent candidate batches: scholarship {sc1} candidates {left}; scholarship {sc2} candidates {right}. For each batch submit only eligible candidates not already applied. Preserve existing records, then finish with direct_answer.')
    eligible_sc3 = [s for s in students if eligibility(kb, s, sc3)['eligible']]
    already_sc3 = [s for s in eligible_sc3 if eligibility(kb, s, sc3)['already_applied']]
    ineligible_sc3 = [s for s in students if s not in eligible_sc3]
    no_op = already_sc3[:1] + ineligible_sc3[:2]
    add_state('verified_noop_submission', [(sc3, no_op)],
        f'Check scholarship {sc3} candidates {no_op}. Make no duplicate or ineligible submission; if none is pending, leave applications unchanged. Preserve existing records, then finish with direct_answer.')
    sc4 = scholarships[4]
    pending = [s for s in students if eligibility(kb, s, sc4)['eligible'] and
               not eligibility(kb, s, sc4)['already_applied']][:2]
    add_state('explicit_pending_pair_submission', [(sc4, pending)],
        f'For scholarship {sc4}, verify exactly the candidate pair {pending} and submit each one only if eligible and not already applied. Do not process anyone else. Preserve existing records, then finish with direct_answer.')
    if Counter(task['difficulty'] for task in tasks) != {'L1': 4, 'L2': 4, 'L3': 4}:
        raise AssertionError('Hold-out task balance changed')
    return tasks


def outcome(kb, task):
    if task['difficulty'] == 'L3':
        return task['final_assertions'][0]['expected']
    spec = task['evaluation']
    if spec['kind'] == 'lookup':
        return QueryEngine(kb).lookup(spec['entity'], spec['field'])['value']
    if spec['kind'] == 'relational':
        return matching_students(kb, spec['query'])
    return _query_answer(kb, spec['query'], count=spec['kind'] == 'count')


_ENTITY = re.compile(r'\b(?:h?sc\d+|h?s\d+|h?c\d+|h?e\d+(?:-\d+)?)\b')
_NUMBER = re.compile(r'\b\d+(?:\.\d+)?\b')


def template_signature(question):
    normalized = _ENTITY.sub('<ENTITY>', question.casefold())
    return _NUMBER.sub('<NUMBER>', normalized)


def entity_ids(kb):
    return {identifier for rows in kb.values() for identifier in rows}


def build_dataset():
    return [(seed, generate_world(seed), None) for seed in HOLDOUT_SEEDS]


def audit_overlap(holdout, references):
    reference_instances, reference_questions, reference_templates = set(), set(), set()
    reference_entities, reference_answers = set(), set()
    for _, kb, tasks in references:
        reference_entities.update(entity_ids(kb))
        for task in tasks:
            reference_instances.add(instance_fingerprint(kb, task))
            reference_questions.add(task['question'])
            reference_templates.add(template_signature(task['question']))
            reference_answers.add(digest(outcome(kb, task)))
    rows = [(kb, task) for _, kb, tasks in holdout for task in tasks]
    holdout_entities = set().union(*(entity_ids(kb) for _, kb, _ in holdout))
    counts = dict(
        exact_instance_overlap=sum(instance_fingerprint(kb, task) in reference_instances for kb, task in rows),
        question_overlap=sum(task['question'] in reference_questions for kb, task in rows),
        template_overlap=sum(template_signature(task['question']) in reference_templates for kb, task in rows),
        answer_overlap=sum(digest(outcome(kb, task)) in reference_answers for kb, task in rows),
        entity_overlap=len(holdout_entities & reference_entities),
    )
    counts.update(holdout_tasks=len(rows), holdout_entities=len(holdout_entities),
        reference_entities=len(reference_entities), definitions=dict(
            exact_instance='SHA-256 of initial state plus task excluding task_id',
            question='exact question string', template='case-folded question with entity IDs and numbers replaced',
            entity='exact ID appearing in any world table', answer='SHA-256 of canonical expected outcome'))
    return counts


def load_frozen_holdout(root, config):
    root = Path(root)
    fixed = dict(version=1, split='test', generator_version=GENERATOR_VERSION,
        data_root='data/grpo_v2_holdout_v1', seeds=list(HOLDOUT_SEEDS), task_count=72,
        tasks_by_difficulty=dict(L1=24, L2=24, L3=24), usage_policy='report_only_never_select')
    if any(config.get(key) != value for key, value in fixed.items()) or set(config) != set(fixed) | {
            'generator_source_sha256', 'manifest_sha256', 'task_list_sha256'}:
        raise ValueError('Frozen hold-out config changed')
    directory = root / config['data_root']
    manifest = json.loads((directory / 'manifest.json').read_text())
    task_list = json.loads((directory / 'task_list.json').read_text())
    if digest(manifest) != config['manifest_sha256'] or digest(task_list) != config['task_list_sha256']:
        raise ValueError('Frozen hold-out manifest or task list changed')
    source_hash = hashlib.sha256((root / 'src/training/grpo_v2/holdout.py').read_bytes()).hexdigest()
    if source_hash != config['generator_source_sha256'] or source_hash != manifest['generator_source_sha256']:
        raise ValueError('Frozen hold-out generator changed')
    items = []
    for seed_record in manifest['seeds_manifest']:
        seed = seed_record['seed']
        kb = json.loads((directory / 'test' / str(seed) / 'kb.json').read_text())
        tasks = json.loads((directory / 'test' / str(seed) / 'tasks.json').read_text())
        if digest(kb) != seed_record['kb_sha256'] or digest(tasks) != seed_record['tasks_sha256']:
            raise ValueError('Frozen hold-out world changed')
        items.extend(dict(seed=seed, kb=kb, task=task) for task in tasks)
    if len(items) != config['task_count'] or Counter(item['task']['difficulty'] for item in items) != config['tasks_by_difficulty']:
        raise ValueError('Frozen hold-out size/balance changed')
    return items, manifest
