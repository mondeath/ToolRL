"""Fixed, uniformly spread 240-task baseline with independent full dev worlds."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .data import read_json, validate_config as validate_smoke, verify_frozen
from src.env_v1.generation import SPLIT_SEEDS
from src.evaluation.generator_audit import instance_fingerprint
from src.evaluation.model_scale import digest, select_tasks


def validate_config(root, config):
    smoke = read_json(Path(root) / 'configs/grpo_v1_smoke.json')
    if digest(smoke) != config['smoke_config_sha256']:
        raise ValueError('Frozen smoke configuration changed')
    for field in ('model_name', 'revision', 'seed', 'group_size', 'rollout', 'sampling', 'lora',
                  'quantization', 'data_manifest_sha256', 'generator_source_sha256', 'diagnostic_tasks_sha256'):
        if config[field] != smoke[field]:
            raise ValueError('Baseline must retain smoke hyperparameters/contract: ' + field)
    expected_training = dict(smoke['training'], max_updates=None)
    if config['training'] != expected_training:
        raise ValueError('No hyperparameter changes or optimizer-step cap allowed')
    if (config['mode'] != 'task_only_baseline' or config['task_count'] != 240
            or config['checkpoint_groups'] != [0, 60, 120, 180, 240] or config['dev_split'] != 'all_400'):
        raise ValueError('Baseline requires one 240-group pass, full dev, and fixed quarter checkpoints')
    compatibility = deepcopy(config)
    compatibility['task_count'] = 24
    compatibility['training']['max_updates'] = 3
    validate_smoke(compatibility)


def load_baseline_data(root, config):
    root = Path(root)
    validate_config(root, config)
    diag_kb, diag_all = verify_frozen(root, config)
    selection = read_json(root / config['tasks_file'])
    if digest(selection) != config['tasks_sha256']:
        raise ValueError('Fixed baseline selection hash mismatch')
    manifest = read_json(root / config['data_root'] / 'manifest.json')
    if digest(manifest) != config['data_manifest_sha256']:
        raise ValueError('Frozen data manifest changed')
    if manifest['split_seeds'] != {k: list(v) for k, v in SPLIT_SEEDS.items()}:
        raise ValueError('Frozen split seeds changed')
    worlds = {}
    def world(split, seed):
        if seed not in SPLIT_SEEDS[split]:
            raise ValueError('Seed not owned by split')
        if (split, seed) not in worlds:
            directory = root / config['data_root'] / split / str(seed)
            kb, tasks = read_json(directory / 'kb.json'), read_json(directory / 'tasks.json')
            original = next(w for w in manifest['seeds'] if w['split'] == split and w['seed'] == seed)
            if digest(kb) != original['kb_sha256'] or digest(tasks) != original['tasks_sha256']:
                raise ValueError('Persisted data changed')
            worlds[(split, seed)] = kb, {task['task_id']: task for task in tasks}
        return worlds[(split, seed)]
    dev = []
    for seed in SPLIT_SEEDS['dev']:
        kb, indexed = world('dev', seed)
        dev.extend(dict(seed=seed, kb=kb, task=task) for task in indexed.values())
    eval_instances = {instance_fingerprint(x['kb'], x['task']) for x in dev}
    eval_instances.update(instance_fingerprint(diag_kb, task) for task in diag_all)
    eval_ids = {x['task']['task_id'] for x in dev} | {task['task_id'] for task in diag_all}
    train = []
    for entry in selection['tasks']:
        if entry['split'] != 'train' or type(entry['seed']) is not int or entry['seed'] not in SPLIT_SEEDS['train']:
            raise ValueError('Non-train task in baseline selection')
        kb, tasks = world('train', entry['seed'])
        task = tasks[entry['task_id']]
        fingerprint = instance_fingerprint(kb, task)
        if fingerprint != entry['instance_sha256'] or task['difficulty'] != entry['difficulty']:
            raise ValueError('Selected instance mismatch')
        if fingerprint in eval_instances or task['task_id'] in eval_ids:
            raise ValueError('Train/eval instance overlap')
        train.append(dict(seed=entry['seed'], kb=kb, task=task, instance_sha256=fingerprint))
    if (len(train) != 240 or len({x['instance_sha256'] for x in train}) != 240
            or Counter(x['seed'] for x in train) != {s: 8 for s in SPLIT_SEEDS['train']}
            or Counter(x['task']['difficulty'] for x in train) != {'L1': 80, 'L2': 80, 'L3': 80}):
        raise ValueError('Require 240 unique tasks, 80 per level and 8 per train seed')
    for start in range(0, 240, 60):
        if Counter(x['seed'] for x in train[start:start + 60]) != {s: 2 for s in SPLIT_SEEDS['train']}:
            raise ValueError('Each quarter must cover every train world twice')
    diag_selection = read_json(root / 'configs/env_v1_diagnostic_tasks.json')
    if digest(diag_selection) != config['diagnostic_tasks_sha256']:
        raise ValueError('Frozen diagnostic selection changed')
    diagnostic = [dict(seed=42, kb=diag_kb, task=t) for t in select_tasks(diag_all, diag_selection['task_ids'])]
    if len(dev) != 400 or len(diagnostic) != 15:
        raise ValueError('Evaluation sizes changed')
    return train, dev, diagnostic


def preflight_data(root, config):
    train, _, diagnostic = load_baseline_data(root, config)
    return train, diagnostic


def dry_run(root, config):
    train, dev, diagnostic = load_baseline_data(root, config)
    # This is a planning interval, not measured throughput. Update after reading
    # actual smoke timings; never label an assumed interval as RTX 3090 evidence.
    group_seconds, eval_seconds = [45, 180], [3, 12]
    eval_episodes = 5 * len(dev) + len(diagnostic)
    return dict(status='passed', training_started=False, selection_sha256=config['tasks_sha256'],
        attempted_groups=len(train), maximum_optimizer_steps=len(train), training_rollouts=len(train) * 4,
        tasks_by_difficulty=dict(Counter(x['task']['difficulty'] for x in train)),
        tasks_by_seed=dict(Counter(x['seed'] for x in train)), group_size=4, passes=1,
        checkpoint_attempted_groups=config['checkpoint_groups'], dev_tasks=len(dev), diagnostic_tasks=len(diagnostic),
        evaluation_schedule=['base: dev400', '25%: dev400', '50%: dev400', '75%: dev400', 'final: dev400 + diagnostic15'],
        evaluation_rollouts=eval_episodes, total_rollouts=4 * len(train) + eval_episodes,
        maximum_training_generation_calls=4 * len(train) * 9 * 2,
        expected_optimizer_steps_if_smoke_rate_holds=round(240 * 0.417),
        estimate=dict(kind='uncalibrated planning assumptions, not GPU measurement',
            seconds_per_training_group=group_seconds, seconds_per_greedy_episode=eval_seconds,
            wall_clock_hours=[round((240 * g + eval_episodes * e) / 3600, 2) for g, e in zip(group_seconds, eval_seconds)],
            excludes='model loading, checkpoint disk IO and network; sequence lengths and GPU contention can exceed range'),
        train_eval_overlap=False, frozen_contract_verified=True)
