"""Generate and audit isolated train/dev worlds. No model loading or training."""
from _common import ROOT
import argparse
import json
from pathlib import Path

from run_model_scale import load_contract
from src.env_v1.generation import GENERATOR_VERSION, SPLIT_SEEDS, generate_instance
from src.evaluation.generator_audit import audit_splits
from src.evaluation.model_scale import digest


def build(output):
    # Verify the existing hashes, including every frozen behavior source.
    load_contract()
    datasets = [(split, seed, *generate_instance(seed, split))
                for split in ('diagnostic', 'train', 'dev') for seed in SPLIT_SEEDS[split]]
    audits = audit_splits(datasets)
    changed_fields = {}
    paths = {'student_gpa': ('students', 'gpa'), 'student_year': ('students', 'year'),
             'student_department': ('students', 'department'),
             'enrollment_course': ('enrollments', 'course_id'),
             'enrollment_completion': ('enrollments', 'status'),
             'scholarship_min_gpa': ('scholarships', 'min_gpa'),
             'scholarship_eligible_year': ('scholarships', 'eligible_year'),
             'scholarship_required_course': ('scholarships', 'required_course')}
    for label, (table, field) in paths.items():
        changed_fields[label] = len({digest({key: row[field] for key, row in kb[table].items()})
                                    for split, seed, kb, tasks in datasets if split != 'diagnostic'})
    changed_fields['initial_applications'] = len({digest(kb['applications'])
        for split, seed, kb, tasks in datasets if split != 'diagnostic'})
    if any(count < 2 for count in changed_fields.values()):
        raise ValueError('A requested data field did not vary')
    outcome_vectors = {level: len({digest([t['outcome_sha256'] for t in audit['tasks']
        if t['difficulty'] == level]) for audit in audits}) for level in ('L1', 'L2', 'L3')}
    if any(count < 2 for count in outcome_vectors.values()):
        raise ValueError('Task answer/final-state outcomes did not vary')
    # Isolate each scholarship's actual eligibility and pending set from
    # unrelated application records in the full final-state assertions.
    from src.env_v1 import eligibility
    from src.env_v1.tasks import initial_state
    target_vectors = set()
    for split, seed, kb, tasks in datasets:
        targets = []
        for task in tasks:
            if task['difficulty'] != 'L3':
                continue
            state = initial_state(kb, task)
            checks = [eligibility(state, sid, task['evaluation']['scholarship_id'])
                      for sid in sorted(state['students'])]
            targets.append([(c['student_id'], c['already_applied']) for c in checks if c['eligible']])
        target_vectors.add(digest(targets))
    if len(target_vectors) < 2:
        raise ValueError('L3 target application sets did not vary')
    report = dict(generator_version=GENERATOR_VERSION, status='passed',
        split_seeds={key: list(value) for key, value in SPLIT_SEEDS.items()},
        frozen_contract_verified=True, reference_paths_checked=sum(len(a['tasks']) for a in audits),
        split_task_counts={split: sum(len(a['tasks']) for a in audits if a['split'] == split)
                           for split in SPLIT_SEEDS},
        unique_kb_hashes=len({a['kb_sha256'] for a in audits}),
        unique_non_id_signatures=len({a['non_id_signature_sha256'] for a in audits}),
        requested_field_distinct_vectors=changed_fields, unique_outcome_vectors=outcome_vectors,
        unique_L3_target_vectors=len(target_vectors), cross_split_instance_duplicates=0,
        instance_definition='task excluding task_id + actual initial KB; templates/individual answers may overlap',
        reference_tool_calls={'L1': 1, 'L2': 3, 'L3': 5}, seeds=audits)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split, seed, kb, tasks in datasets:
        if split == 'diagnostic':
            continue  # Do not rewrite frozen diagnostic files.
        directory = output / split / str(seed)
        directory.mkdir(parents=True)
        for name, value in [('kb.json', kb), ('tasks.json', tasks)]:
            (directory / name).write_text(json.dumps(value, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    summary = {key: value for key, value in report.items() if key != 'seeds'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/grpo_v1_data')
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error('Output directory already exists; choose a new path to preserve prior artifacts')
    print(json.dumps(build(args.output_root), indent=2))


if __name__ == '__main__':
    main()
