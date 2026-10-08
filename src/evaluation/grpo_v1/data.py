"""Fixed held-out selection and explicit seed/entity/template/answer audit."""
from collections import Counter
import random
import re

from src.env_v1.generation import generate_instance, SPLIT_SEEDS
from src.env_v1.tasks import initial_state
from src.env_v1 import matching_students, eligibility
from src.tools import QueryEngine
from src.evaluation.model_scale import digest
from src.evaluation.generator_audit import instance_fingerprint
from src.training.grpo_v1.data import read_json, verify_frozen


SELECTION_SEED = 20261008


def expected_outcome(kb, task):
    state = initial_state(kb, task)
    spec = task['evaluation']
    engine = QueryEngine(state)
    if spec['kind'] == 'state':
        sc = spec['scholarship_id']
        checks = [eligibility(state, sid, sc) for sid in sorted(state['students'])]
        # Compare the relevant final target and missing submissions, not just
        # a full-KB hash that could hide identical subproblems in other worlds.
        return dict(scholarship_id=sc, eligible=[c['student_id'] for c in checks if c['eligible']],
                    pending=[c['student_id'] for c in checks if c['eligible'] and not c['already_applied']])
    if spec['kind'] == 'lookup':
        return engine.lookup(spec['entity'], spec['field'])['value']
    if spec['kind'] == 'relational':
        return matching_students(state, spec['query'])
    result = engine.execute(spec['query'])
    return result['matched_count'] if spec['kind'] == 'count' else result['answer']


def template(task):
    text = re.sub(r'\b(?:sc|s|c|e)\d+(?:-\d+)?\b', '<ID>', task['question'])
    return re.sub(r'\b\d+(?:\.\d+)?\b', '<NUM>', text)


def task_entities(task):
    return set(re.findall(r'\b(?:sc|s|c|e)\d+(?:-\d+)?\b', task['question']))


def prepare_selection(root, run_dir):
    config = read_json(run_dir / 'config.json')
    verify_frozen(root, config)
    actual = read_json(run_dir / 'tasks.json')
    if digest(actual) != config['tasks_sha256']:
        raise ValueError('Archived training task selection/hash mismatch')
    records = []
    rng = random.Random(SELECTION_SEED)
    for seed in SPLIT_SEEDS['dev']:
        kb, tasks = generate_instance(seed, 'dev')
        pools = {}
        for level in ('L1', 'L2', 'L3'):
            pool = [t for t in tasks if t['difficulty'] == level]
            rng.shuffle(pool)
            pools[level] = pool[:2]
        for index in range(2):
            for level in ('L1', 'L2', 'L3'):
                task = pools[level][index]
                records.append(dict(split='dev', seed=seed, task_id=task['task_id'], difficulty=level,
                                    instance_sha256=instance_fingerprint(kb, task)))
    body = dict(version=1,selection_seed=SELECTION_SEED, training_selection_sha256=digest(actual),
                provenance='retrospective dev holdout, seen in intermediate eval, never used for gradient training', tasks=records)
    return dict(**body, selection_sha256=digest(body))


def load_selection(root, run_dir, selection):
    body = {k: v for k, v in selection.items() if k != 'selection_sha256'}
    if digest(body) != selection['selection_sha256']:
        raise ValueError('Evaluation selection hash mismatch')
    expected = prepare_selection(root, run_dir)
    if selection != expected:
        raise ValueError('Evaluation list differs from predeclared outcome-independent selection')
    def resolve(entry):
        kb, tasks = generate_instance(entry['seed'], entry['split'])
        task = next(t for t in tasks if t['task_id'] == entry['task_id'])
        if instance_fingerprint(kb, task) != entry['instance_sha256']:
            raise ValueError('Task instance hash mismatch')
        return dict(kb=kb, task=task, seed=entry['seed'])
    train_entries = read_json(run_dir / 'tasks.json')['tasks']
    train = [resolve(entry) for entry in train_entries]
    evaluation = [resolve(entry) for entry in selection['tasks']]
    if len(evaluation) != 60 or Counter(x['task']['difficulty'] for x in evaluation) != {'L1': 20, 'L2': 20, 'L3': 20}:
        raise ValueError('Require 60 tasks, 20 per difficulty')
    return train, evaluation


def leakage_audit(train, evaluation):
    train_ids = {x['task']['task_id'] for x in train}
    train_seeds = {x['seed'] for x in train}
    train_instances = {instance_fingerprint(x['kb'], x['task']) for x in train}
    templates = {template(x['task']) for x in train}
    questions = {x['task']['question'] for x in train}
    entity_ids = set().union(*(task_entities(x['task']) for x in train))
    answers = {(x['task']['evaluation']['kind'], digest(expected_outcome(x['kb'], x['task']))) for x in train}
    query_answers = {(digest(x['task']['evaluation']), digest(expected_outcome(x['kb'], x['task']))) for x in train}
    details = []
    for item in evaluation:
        task = item['task']
        outcome = expected_outcome(item['kb'], task)
        details.append(dict(task_id=task['task_id'], seed=item['seed'], difficulty=task['difficulty'],
            seed_overlap=item['seed'] in train_seeds, task_id_overlap=task['task_id'] in train_ids,
            exact_instance_overlap=instance_fingerprint(item['kb'], task) in train_instances,
            template_overlap=template(task) in templates, exact_question_overlap=task['question'] in questions,
            referenced_entity_overlap=bool(task_entities(task) & entity_ids),
            expected_answer_overlap=(task['evaluation']['kind'], digest(outcome)) in answers,
            query_and_answer_overlap=(digest(task['evaluation']), digest(outcome)) in query_answers,
            expected_outcome=outcome))
    flags = ['seed_overlap', 'task_id_overlap', 'exact_instance_overlap', 'template_overlap', 'exact_question_overlap',
             'referenced_entity_overlap', 'expected_answer_overlap', 'query_and_answer_overlap']
    counts = {key: sum(row[key] for row in details) for key in flags}
    if any(counts[key] for key in flags[:3]):
        raise ValueError('Training/evaluation split or instance overlap')
    return dict(task_count=len(evaluation), counts=counts, per_task=details,
        already_used_as_dev=True, globally_novel_entity_ids=False,
        interpretation='Seed/full-instance disjointness does not remove reusable templates, entity labels or repeated answers. '
                       'No tasks filtered by model performance or answer novelty. Not a fresh blind test.')


def subset(items, kind):
    if kind == 'full':
        return items
    if kind == 'pilot':
        return [next(x for x in items if x['task']['difficulty'] == level) for level in ('L1', 'L2', 'L3')]
    if kind == 'robustness':
        seeds = sorted({x['seed'] for x in items})[:4]
        return [next(x for x in items if x['seed'] == seed and x['task']['difficulty'] == level)
                for seed in seeds for level in ('L1', 'L2', 'L3')]
    raise ValueError('Unknown subset')
