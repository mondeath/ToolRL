"""Create or verify the frozen V2 hold-out dataset; never touches V1 data."""
from _common import ROOT
import argparse
import hashlib
import json
from pathlib import Path

from src.env_v1 import generate_kb
from src.env_v1.generation import SPLIT_SEEDS, generate_instance
from src.env_v1.tasks import generate_tasks as generate_v1_tasks
from src.evaluation.generator_audit import instance_fingerprint
from src.evaluation.model_scale import digest
from src.training.grpo_v1.data import read_json
from src.training.grpo_v2.holdout import (GENERATOR_VERSION, HOLDOUT_SEEDS, audit_overlap,
    generate_tasks, generate_world, outcome, template_signature)


OUTPUT = ROOT / 'data/grpo_v2_holdout_v1'


def reference_datasets():
    rows = []
    for split in ('train', 'dev'):
        for seed in SPLIT_SEEDS[split]:
            directory = ROOT / 'results/grpo_v1_data' / split / str(seed)
            rows.append((seed, read_json(directory / 'kb.json'), read_json(directory / 'tasks.json')))
    rows.append((42, generate_kb(42), generate_v1_tasks(generate_kb(42))))
    return rows


def payloads():
    holdout, files, task_list, seeds = [], {}, [], []
    for seed in HOLDOUT_SEEDS:
        kb = generate_world(seed)
        tasks = generate_tasks(kb, seed)
        holdout.append((seed, kb, tasks))
        files[f'test/{seed}/kb.json'] = kb
        files[f'test/{seed}/tasks.json'] = tasks
        task_rows = []
        for task in tasks:
            row = dict(seed=seed, task_id=task['task_id'], difficulty=task['difficulty'],
                template_id=task['template_id'], instance_sha256=instance_fingerprint(kb, task),
                question_sha256=hashlib.sha256(task['question'].encode()).hexdigest(),
                template_sha256=hashlib.sha256(template_signature(task['question']).encode()).hexdigest(),
                answer_sha256=digest(outcome(kb, task)))
            task_rows.append(row)
            task_list.append(row)
        seeds.append(dict(seed=seed, kb_sha256=digest(kb), tasks_sha256=digest(tasks), tasks=task_rows))
    files['task_list.json'] = dict(version=1, split='test', generator_version=GENERATOR_VERSION,
        selection_policy='all generated hold-out tasks; forbidden for tuning or checkpoint selection', tasks=task_list)
    source_hash = hashlib.sha256((ROOT / 'src/training/grpo_v2/holdout.py').read_bytes()).hexdigest()
    files['manifest.json'] = dict(version=1, generator_version=GENERATOR_VERSION,
        generator_source_sha256=source_hash, status='frozen', split='test', seeds=list(HOLDOUT_SEEDS),
        task_count=len(task_list), tasks_by_difficulty=dict(L1=24, L2=24, L3=24),
        task_list_sha256=digest(files['task_list.json']), seeds_manifest=seeds,
        overlap_audit=audit_overlap(holdout, reference_datasets()),
        usage_policy='Hold-out is report-only. Never use it for hyperparameter, prompt, checkpoint, or run selection.')
    return files


def encoded(files):
    return {name: (json.dumps(value, indent=2, allow_nan=False) + '\n').encode() for name, value in files.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    expected = encoded(payloads())
    if args.verify:
        for name, data in expected.items():
            path = OUTPUT / name
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit('Frozen hold-out mismatch: ' + str(path))
        print(json.dumps(dict(status='verified', files=len(expected), output=str(OUTPUT))))
        return 0
    if OUTPUT.exists():
        parser.error('output already exists; use --verify instead of overwriting frozen data')
    for name, data in expected.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps(dict(status='created', files=len(expected), output=str(OUTPUT))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
