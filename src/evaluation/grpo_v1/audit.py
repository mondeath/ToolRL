"""Audit actual run identity and PEFT artifacts without trusting a folder label."""
import hashlib
from pathlib import Path

from src.training.grpo_v1.data import read_json


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def audit_run(run_dir, checkpoint=None, expected_model=None):
    run_dir = Path(run_dir)
    config = read_json(run_dir / 'config.json')
    runtime = read_json(run_dir / 'model_runtime.json')
    status = read_json(run_dir / 'status.json')
    records = {str(step): read_json(run_dir / f'checkpoint_group_{step}.json') for step in (180, 240)}
    blockers = []
    for record in records.values():
        if record['config'] != config or record['selection_sha256'] != config['tasks_sha256']:
            blockers.append('Run/checkpoint metadata disagreement')
    if runtime['revision'] != config['revision']:
        blockers.append('Runtime/config revision mismatch')
    if expected_model and config['model_name'] != expected_model:
        blockers.append(f'Expected {expected_model}, but archived run identifies {config["model_name"]}')
    checkpoint = Path(checkpoint) if checkpoint else run_dir / 'checkpoints/group_240'
    files = {name: (checkpoint / name).is_file() for name in
             ('adapter_config.json', 'adapter_model.safetensors', 'metadata.json')}
    tensors = None
    if not all(files.values()):
        blockers.append('Actual PEFT adapter/config/metadata files unavailable; checkpoint_group_240.json is not weights')
    else:
        from safetensors import safe_open
        adapter = read_json(checkpoint / 'adapter_config.json')
        metadata = read_json(checkpoint / 'metadata.json')
        if any(metadata.get(key) != records['240'].get(key) for key in
               ('config', 'selection_sha256', 'attempted_groups', 'optimizer_steps', 'base_digest')):
            blockers.append('Actual checkpoint is not the final checkpoint of this run')
        if adapter.get('peft_type') != 'LORA' or adapter.get('r') != config['lora']['r']:
            blockers.append('Adapter type/rank mismatch')
        if adapter.get('lora_alpha') != config['lora']['lora_alpha'] or set(adapter.get('target_modules', [])) != set(config['lora']['target_modules']):
            blockers.append('Adapter configuration mismatch')
        path_identity = adapter.get('base_model_name_or_path', '')
        if ('Qwen2.5-' in path_identity and config['model_name'].split('/')[-1] not in path_identity):
            blockers.append('Adapter base model path disagrees with run identity')
        with safe_open(str(checkpoint / 'adapter_model.safetensors'), framework='pt', device='cpu') as weights:
            tensors = {key: list(weights.get_slice(key).get_shape()) for key in weights.keys()}
        if not tensors or any('lora_A' not in name and 'lora_B' not in name for name in tensors):
            blockers.append('Unexpected or empty adapter tensor payload')
        if sum(__import__('math').prod(shape) for shape in tensors.values()) != runtime['trainable_params']:
            blockers.append('Adapter tensor count disagrees with measured trainable parameters')
    return dict(model_name_from_artifacts=config['model_name'], revision=config['revision'],
        expected_model=expected_model, total_params=runtime['total_params'], adapter_params=runtime['trainable_params'],
        attempted_groups=status['attempted_groups'], optimizer_steps=status['optimizer_steps'],
        run_reported_checkpoint_reload=status['checkpoint_reload'], independently_restored=False,
        checkpoint_path=str(checkpoint), checkpoint_files=files, adapter_tensor_shapes=tensors,
        restoration_ready=not blockers, blockers=blockers,
        config_sha256=file_hash(run_dir / 'config.json'),
        reward_path='collector.task_reward -> float(original evaluator task_success) -> group_advantages -> clipped_segment_loss',
        cost_penalty_in_training=False, verifier_reward=False, kl_beta=config['training']['beta'],
        note='Repository code and run artifacts audited; missing adapter bytes cannot be independently verified.')
