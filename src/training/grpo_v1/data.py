"""Load the fixed smoke selection; refuse changes to frozen data contracts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.env_v1.generation import generate_instance, SPLIT_SEEDS
from src.env_v1.protocol import ACTION_SCHEMA, SYSTEM_PROMPT
from src.evaluation.generator_audit import instance_fingerprint
from src.evaluation.model_scale import digest, select_tasks


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_config(config):
    if config['model_name'] != 'Qwen/Qwen2.5-7B-Instruct' or config['revision'] != 'a09a35458c702b33eeacc393d103063234e8bc28':
        raise ValueError('Only the frozen Qwen2.5-7B revision is supported')
    if config['group_size'] != 4 or config['task_count'] != 24:
        raise ValueError('Smoke requires exactly 24 tasks and G=4')
    sampling = config['sampling']
    if sampling != dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0,
                        max_new_tokens=256, max_context_tokens=16384):
        raise ValueError('Frozen smoke sampling changed')
    if config['rollout'] != dict(max_steps=9, max_tool_calls=8, max_parse_retries=1, seed=2026):
        raise ValueError('Frozen interaction budget changed')
    train = config['training']
    if (not 1 <= train['max_updates'] <= 3 or train['epochs'] != 1 or train['beta'] != 0
            or train['weight_decay'] != 0 or not train['gradient_checkpointing']
            or train['logprob_space'] != 'raw_model_softmax'):
        raise ValueError('Only bounded adapter-only, zero-KL smoke is supported')
    if not 0 < train['learning_rate'] <= 1e-4 or not 0 < train['clip_epsilon'] < 1 or train['max_grad_norm'] <= 0:
        raise ValueError('Invalid optimizer configuration')
    if config['lora'] != dict(r=16, lora_alpha=32, lora_dropout=0.0, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
        raise ValueError('Frozen adapter configuration changed')
    if config['quantization'] != dict(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype='bfloat16'):
        raise ValueError('NF4/bf16 required')


def verify_frozen(root, config):
    frozen = read_json(root / 'configs/env_v1_model_scale.json')['frozen']
    for name, expected in frozen['source_sha256'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        # The only post-freeze compatibility repair expands the already
        # conservative surface normalizer to the three env_v1 actions that its
        # schema always supported.  Keep the archived V1 hash in its config;
        # accept this one reviewed source hash instead of rewriting history.
        reviewed_normalizer_fix = (name == 'src/policies/normalizer.py' and
            actual == 'f3cf012677be677cab32f9608edc44c69ea5ae01141af5dbd2e349706d81174a')
        if actual != expected and not reviewed_normalizer_fix:
            raise ValueError('Frozen source changed: ' + name)
    if hashlib.sha256((root / 'src/env_v1/generation.py').read_bytes()).hexdigest() != config['generator_source_sha256']:
        raise ValueError('Frozen data generator changed')
    kb, tasks = generate_instance(42, 'diagnostic')
    if digest(kb) != frozen['kb_sha256'] or digest(tasks) != frozen['all_tasks_sha256']:
        raise ValueError('Frozen diagnostic world changed')
    if hashlib.sha256((SYSTEM_PROMPT + json.dumps(ACTION_SCHEMA, sort_keys=True)).encode()).hexdigest() != frozen['prompt_sha256']:
        raise ValueError('Frozen prompt/schema changed')
    return kb, tasks


def load_data(root, config):
    root = Path(root)
    validate_config(config)
    diagnostic_kb, all_diagnostic = verify_frozen(root, config)
    selection = read_json(root / config['tasks_file'])
    if digest(selection) != config['tasks_sha256']:
        raise ValueError('Fixed smoke task list changed')
    manifest = read_json(root / config['data_root'] / 'manifest.json')
    if digest(manifest) != config['data_manifest_sha256']:
        raise ValueError('Frozen data manifest changed')
    if manifest['split_seeds'] != {key: list(value) for key, value in SPLIT_SEEDS.items()}:
        raise ValueError('Split seed ownership changed')
    eval_fingerprints = {instance_fingerprint(diagnostic_kb, task) for task in all_diagnostic}
    eval_ids = {task['task_id'] for task in all_diagnostic}
    for seed in SPLIT_SEEDS['dev']:
        kb, tasks = generate_instance(seed, 'dev')
        eval_fingerprints.update(instance_fingerprint(kb, task) for task in tasks)
        eval_ids.update(task['task_id'] for task in tasks)
    selected, worlds = [], {}
    for entry in selection['tasks']:
        seed = entry['seed']
        if entry['split'] != 'train' or type(seed) is not int or seed not in SPLIT_SEEDS['train']:
            raise ValueError('Training selection contains a non-train seed')
        if seed not in worlds:
            directory = root / config['data_root'] / 'train' / str(seed)
            kb, tasks = read_json(directory / 'kb.json'), read_json(directory / 'tasks.json')
            expected = next(a for a in manifest['seeds'] if a['split'] == 'train' and a['seed'] == seed)
            if digest(kb) != expected['kb_sha256'] or digest(tasks) != expected['tasks_sha256']:
                raise ValueError('Persisted train data changed')
            worlds[seed] = kb, {task['task_id']: task for task in tasks}
        kb, indexed = worlds[seed]
        task = indexed[entry['task_id']]
        fingerprint = instance_fingerprint(kb, task)
        if fingerprint != entry['instance_sha256'] or task['difficulty'] != entry['difficulty']:
            raise ValueError('Selected instance changed')
        if fingerprint in eval_fingerprints or task['task_id'] in eval_ids:
            raise ValueError('Train/eval overlap')
        selected.append(dict(seed=seed, kb=kb, task=task, instance_sha256=fingerprint))
    if (len(selected) != 24 or Counter(x['task']['difficulty'] for x in selected) != {'L1': 8, 'L2': 8, 'L3': 8}
            or len({x['seed'] for x in selected}) != 8
            or len({x['instance_sha256'] for x in selected}) != 24
            or len({x['task']['task_id'] for x in selected}) != 24):
        raise ValueError('Require 24 distinct tasks, 8 per level, across 8 seeds')
    diagnostic_selection = read_json(root / 'configs/env_v1_diagnostic_tasks.json')
    if digest(diagnostic_selection) != config['diagnostic_tasks_sha256']:
        raise ValueError('Frozen diagnostic selection changed')
    ids = diagnostic_selection['task_ids']
    diagnostic = [dict(seed=42, kb=diagnostic_kb, task=task) for task in select_tasks(all_diagnostic, ids)]
    if len(diagnostic) != 15:
        raise ValueError('Diagnostic subset changed')
    return selected, diagnostic
