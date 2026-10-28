"""V2 config validation while reusing the immutable V1 task/data selection."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from src.evaluation.model_scale import digest
from src.training.grpo_v1.baseline_data import load_baseline_data
from src.training.grpo_v1.data import load_data as load_v1_smoke_data
from src.training.grpo_v1.data import read_json
from .alignment import validate_sampling_logprob_alignment


def validate_config(root, config):
    root = Path(root)
    v1 = read_json(root / 'configs/grpo_v1_baseline.json')
    if digest(v1) != config['v1_source_config_sha256']:
        raise ValueError('Frozen V1 source config changed')
    if config.get('version') != 2 or config.get('mode') != 'grpo_v2_task_only':
        raise ValueError('Require GRPO V2 task-only mode')
    if config.get('output_namespace') != 'results/grpo_v2':
        raise ValueError('V2 outputs must use the independent results/grpo_v2 namespace')
    if not config.get('enforce_sampling_logprob_alignment'):
        raise ValueError('V2 must enforce sampling/log-prob alignment')
    validate_sampling_logprob_alignment(config['sampling'], config['training']['logprob_space'])
    for field in ('model_name', 'revision', 'data_root', 'tasks_file', 'tasks_sha256',
                  'data_manifest_sha256', 'generator_source_sha256', 'group_size', 'task_count',
                  'seed', 'rollout', 'lora', 'quantization', 'diagnostic_tasks_sha256',
                  'checkpoint_groups', 'dev_split', 'selection_seed'):
        if config[field] != v1[field]:
            raise ValueError('V2 must retain the frozen V1 data/model contract: ' + field)
    if config['training'] != v1['training']:
        raise ValueError('V2 keeps the V1 single-update optimizer contract unchanged')
    limitation = config.get('known_limitations', {}).get('single_update_clipping')
    if not limitation or 'clipping' not in limitation.lower():
        raise ValueError('V2 must record the single-update clipping limitation')
    return validate_sampling_logprob_alignment(config['sampling'], config['training']['logprob_space'])


def load_v2_data(root, config):
    validate_config(root, config)
    # Task worlds and selection stay byte-for-byte tied to V1; only the
    # behavior-policy sampling contract changes in V2.
    v1 = read_json(Path(root) / 'configs/grpo_v1_baseline.json')
    return load_baseline_data(root, v1)


def validate_smoke_config(root, config):
    root = Path(root)
    baseline = read_json(root / 'configs/grpo_v2_baseline.json')
    if digest(baseline) != config['v2_baseline_config_sha256']:
        raise ValueError('Frozen V2 baseline config changed')
    if config.get('version') != 2 or config.get('mode') != 'grpo_v2_gpu_smoke':
        raise ValueError('Require GRPO V2 GPU smoke mode')
    for field in ('model_name', 'revision', 'data_root', 'data_manifest_sha256',
                  'generator_source_sha256', 'group_size', 'seed', 'rollout', 'sampling',
                  'lora', 'quantization', 'training', 'diagnostic_tasks_sha256',
                  'selection_seed', 'output_namespace', 'enforce_sampling_logprob_alignment',
                  'known_limitations'):
        if config[field] != baseline[field]:
            raise ValueError('Smoke must retain the V2 baseline contract: ' + field)
    expected = dict(task_count=24, tasks_file='configs/grpo_v1_smoke_tasks.json',
        tasks_sha256='ba3cc2a0c3099ff97a5de1184727da790529754d396c9438b31d4f26779966f8',
        checkpoint_groups=[0, 24], dev_split='diagnostic15_smoke_only')
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError('Invalid V2 smoke setting: ' + field)
    if not config.get('smoke_only') or config.get('holdout_evaluation_enabled'):
        raise ValueError('Smoke must be explicitly bounded and must not evaluate hold-out')
    return validate_sampling_logprob_alignment(config['sampling'], config['training']['logprob_space'])


def load_v2_smoke_data(root, config):
    validate_smoke_config(root, config)
    v1_smoke = read_json(Path(root) / 'configs/grpo_v1_smoke.json')
    train, diagnostic = load_v1_smoke_data(root, v1_smoke)
    if len(train) != 24 or len(diagnostic) != 15:
        raise ValueError('Unexpected smoke train/diagnostic size')
    return train, diagnostic, diagnostic


def smoke_preflight_data(root, config):
    train, _, diagnostic = load_v2_smoke_data(root, config)
    return train, diagnostic


def preflight_data(root, config):
    train, _, diagnostic = load_v2_data(root, config)
    return train, diagnostic


def dry_run(root, config):
    train, dev, diagnostic = load_v2_data(root, config)
    return dict(status='passed', training_started=False, version=2,
        output_namespace=config['output_namespace'], selection_sha256=config['tasks_sha256'],
        attempted_groups=len(train), maximum_optimizer_steps=len(train),
        training_rollouts=len(train) * config['group_size'], group_size=config['group_size'], passes=1,
        tasks_by_difficulty=dict(Counter(x['task']['difficulty'] for x in train)),
        tasks_by_seed=dict(Counter(x['seed'] for x in train)),
        checkpoint_attempted_groups=config['checkpoint_groups'], dev_tasks=len(dev),
        diagnostic_tasks=len(diagnostic), aligned_sampling=validate_config(root, config),
        limitation=config['known_limitations']['single_update_clipping'],
        train_eval_overlap=False, frozen_v1_task_contract_reused=True)


def smoke_dry_run(root, config):
    train, dev, diagnostic = load_v2_smoke_data(root, config)
    return dict(status='passed', training_started=False, version=2, mode=config['mode'], smoke_only=True,
        output_namespace=config['output_namespace'], selection_sha256=config['tasks_sha256'],
        attempted_groups=len(train), maximum_optimizer_steps=len(train),
        training_rollouts=len(train) * config['group_size'], group_size=config['group_size'], passes=1,
        tasks_by_difficulty=dict(Counter(x['task']['difficulty'] for x in train)),
        tasks_by_seed=dict(Counter(x['seed'] for x in train)),
        checkpoint_attempted_groups=config['checkpoint_groups'], dev_tasks=len(dev),
        diagnostic_tasks=len(diagnostic), aligned_sampling=validate_smoke_config(root, config),
        evaluation_schedule=['base: diagnostic15', 'final: diagnostic15', 'final: diagnostic15 integrity repeat'],
        holdout_evaluation_enabled=False, limitation=config['known_limitations']['single_update_clipping'],
        train_eval_overlap=False, frozen_v1_smoke_selection_reused=True)
