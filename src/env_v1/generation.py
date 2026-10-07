"""Seeded data expansion only; frozen seed-42 generator and runtime stay intact."""
import random

from . import generate_kb as frozen_generate_kb
from .tasks import generate_tasks as frozen_generate_tasks

GENERATOR_VERSION = 'env-v1-instances-1'
SPLIT_SEEDS = {'train': tuple(range(1000, 1030)),
               'dev': tuple(range(2000, 2010)), 'diagnostic': (42,)}


def generate_instance(seed, split):
    """Return (KB, full task set); split ownership is checked before generation.

    Keep IDs meaningful and fixed across seeds. Novelty comes from facts, not
    renaming. Namespace task IDs only to make dataset provenance unambiguous.
    """
    if type(seed) is not int or split not in SPLIT_SEEDS or seed not in SPLIT_SEEDS[split]:
        raise ValueError('Seed is not allocated to this split')
    if split == 'diagnostic':
        kb = frozen_generate_kb(42)
        return kb, frozen_generate_tasks(kb)

    rng = random.Random(seed)
    kb = frozen_generate_kb(42)
    kb['enrollments'], kb['applications'] = {}, {}
    departments = ['Logic', 'Robotics', 'Ecology']
    core_courses = rng.sample(list(kb['courses']), 5)
    filler_courses = [cid for cid in kb['courses'] if cid not in core_courses]
    years = [1, 2, 3, 4, rng.randint(1, 4)]
    rng.shuffle(years)
    for course in kb['courses'].values():
        course.update(department=rng.choice(departments), credits=rng.choice([2, 3, 4]))

    for group in range(5):
        sc, course, year = f'sc{group:02}', core_courses[group], years[group]
        min_gpa = rng.randint(330, 360) / 100
        kb['scholarships'][sc].update(min_gpa=min_gpa, eligible_year=year, required_course=course)
        # Three eligible students; three independent failure cases. Shuffle
        # which fixed IDs have those roles, but also vary numeric/relational facts.
        members = [f's{group * 6 + j:02}' for j in range(6)]
        rng.shuffle(members)
        for role, sid in enumerate(members):
            if role == 2:
                # Preserve the original 3-vs-2 distinction between the fixed
                # L2 thresholds, as well as three scholarship-eligible members.
                gpa = rng.randint(max(350, round(min_gpa * 100)), 364) / 100
            elif role == 3:
                gpa = rng.randint(270, min(349, round(min_gpa * 100) - 1)) / 100
            else:
                gpa = rng.randint(370, 400) / 100
            student_year = rng.choice([y for y in range(1, 5) if y != year]) if role == 5 else year
            kb['students'][sid].update(gpa=gpa, year=student_year, department=rng.choice(departments))
            for j, cid in enumerate([course] + rng.sample(filler_courses, 2)):
                status = ('enrolled' if role == 4 else 'completed') if j == 0 else rng.choice(['enrolled', 'completed'])
                eid = f'e{int(sid[1:]):02}-{j}'
                kb['enrollments'][eid] = dict(id=eid, student_id=sid, course_id=cid,
                    status=status, grade=rng.choice(['A', 'B', 'C']) if status == 'completed' else None)
        for sid in rng.sample(members[:3], rng.choice([1, 2])):
            aid = f'{sid}:{sc}'
            kb['applications'][aid] = dict(id=aid, student_id=sid, scholarship_id=sc, status='submitted')

    tasks = frozen_generate_tasks(kb)
    for task in tasks:
        spec = task['evaluation']
        # Same year/GPA/completed-course intersection task, with the new cohort
        # facts. Thresholds and query structure are unchanged (3.5 and 3.65).
        if spec['kind'] == 'relational' and 'course_id' in spec['query']:
            query = spec['query']
            group = int(query['course_id'][1:])
            query.update(year=years[group], course_id=core_courses[group])
            task['question'] = (f'List all year {query["year"]} student IDs with GPA at least '
                                f'{query["min_gpa"]} who completed course {query["course_id"]}.')
            task['prompt'] = task['question']
        task['task_id'] = f'{split}-{seed}-{task["task_id"]}'
    return kb, tasks
