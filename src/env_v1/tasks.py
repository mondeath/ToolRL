"""Offline task specifications; public_task still exposes only task_id/question."""
from copy import deepcopy
from . import matching_students


def initial_state(kb, task):
    state = deepcopy(kb)
    if 'initial_applications' in task:
        state['applications'] = deepcopy(task['initial_applications'])
    return state


def generate_tasks(kb):
    tasks = []
    def add(level, question, spec, **extra):
        tasks.append(dict(task_id=f'v1-{level}-{sum(t["difficulty"] == level for t in tasks):02}',
                          difficulty=level, question=question, prompt=question, evaluation=spec,
                          final_assertions=[], forbidden_assertions=[], failure_profile=None, **extra))
    for i in range(5):
        sid = f's{i:02}'
        add('L1', f'What is the GPA of student {sid}? Return a number.', dict(kind='lookup', entity=sid, field='gpa'))
    for year in range(1, 5):
        add('L1', f'List all student IDs in year {year}.', dict(kind='search', query=dict(table='students', where=[dict(field='year', value=year)])))
    add('L1', 'How many courses have 3 credits? Return a number.', dict(kind='count', query=dict(table='courses', where=[dict(field='credits', value=3)])))
    for group in range(5):
        sc = f'sc{group:02}'
        add('L2', f'List all student IDs eligible for scholarship {sc}, including those already applied. Check year, GPA and completed required course.', dict(kind='relational', query=dict(scholarship_id=sc)))
        for threshold in (3.5, 3.65):
            year, cid = group % 4 + 1, f'c{group:02}'
            add('L2', f'List all year {year} student IDs with GPA at least {threshold} who completed course {cid}.',
                dict(kind='relational', query=dict(year=year, min_gpa=threshold, course_id=cid)))
    for group in range(5):
        sc = f'sc{group:02}'
        eligible = matching_students(kb, dict(scholarship_id=sc))
        for existing in eligible:
            apps = {a: deepcopy(row) for a, row in kb['applications'].items() if row['scholarship_id'] != sc}
            aid = f'{existing}:{sc}'
            apps[aid] = dict(id=aid, student_id=existing, scholarship_id=sc, status='submitted')
            add('L3', f'Submit scholarship {sc} applications for every eligible student who has not already applied. Do not submit duplicates or ineligible students. Preserve all existing records. Then finish with direct_answer.',
                dict(kind='state', scholarship_id=sc), initial_applications=apps)
            target = deepcopy(apps)
            for sid in eligible:
                aid = f'{sid}:{sc}'
                target[aid] = dict(id=aid, student_id=sid, scholarship_id=sc, status='submitted')
            tasks[-1]['final_assertions'] = [dict(type='applications_equal', expected=target)]
            tasks[-1]['forbidden_assertions'] = [dict(type='non_application_mutation'), dict(type='existing_application_mutation')]
    return tasks
