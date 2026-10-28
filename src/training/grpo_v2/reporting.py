"""V2 report and complete artifact manifest generation."""
import hashlib
import json
from pathlib import Path


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def render_report(directory):
    directory = Path(directory)
    status = _read(directory / 'status.json') or {'status': 'not_run'}
    summary = _read(directory / 'training_summary.json') or {}
    lines = ['# GRPO V2 training report', '', f'Status: {status["status"]}.', '',
        'Behavior sampling is aligned with stored raw model-softmax log-probabilities: '
        '`do_sample=true, temperature=1.0, top_p=1.0, top_k=0`.', '',
        'This remains a single on-policy update per mixed-reward group. Because old and current '
        'log-probabilities are evaluated before that sole update, ratios should begin near 1 and '
        'clipping is expected to be largely inactive; V2 intentionally does not add multi-epoch updates.', '',
        f'Attempted groups: {summary.get("attempted_groups", 0)}',
        f'Optimizer steps: {summary.get("optimizer_steps", 0)}',
        f'Zero-variance groups: {summary.get("zero_variance_groups", 0)}',
        f'Mean reward variance: {summary.get("mean_reward_variance")}',
        f'Mean ratio: {summary.get("ratio_mean")}',
        f'Clip fraction: {summary.get("clip_fraction")}',
        f'Approx KL: {summary.get("approx_kl")}', '',
        'No GPU reproduction claim is made by CPU tests or dry-run validation.', '']
    (directory / 'GRPO_V2_RESULTS.md').write_text('\n'.join(lines))
    return '\n'.join(lines)


def write_manifest(directory, config, train_items):
    directory = Path(directory)
    artifacts = []
    for path in sorted(p for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'):
        relative = str(path.relative_to(directory))
        artifacts.append(dict(path=relative, size_bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(schema_version=1, experiment_version='grpo_v2', status=_read(directory / 'status.json'),
        config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        task_selection_sha256=config['tasks_sha256'], data_manifest_sha256=config['data_manifest_sha256'],
        random_seeds=dict(global_seed=config['seed'], rollout_seed=config['rollout']['seed'],
            selection_seed=config['selection_seed'], train_world_seeds=sorted({x['seed'] for x in train_items}),
            session_seed_derivation='sha256(base_seed:task_id:rollout_id), then sha256(session_seed:generation_index)'),
        required_artifacts=dict(config='config.json', tasks='tasks.json', training_rollouts='rollouts.jsonl',
            training_metrics='metrics.jsonl', adapters_and_checkpoints='checkpoints/',
            raw_evaluation_trajectories=['base_dev/results.jsonl', 'eval_group_*/dev/results.jsonl',
                f'eval_group_{config["task_count"]:03}/diagnostic/results.jsonl']),
        limitation=config['known_limitations']['single_update_clipping'], artifacts=artifacts)
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    temporary.replace(directory / 'manifest.json')
    return manifest


def validate_artifact_manifest(directory):
    directory = Path(directory)
    manifest = _read(directory / 'manifest.json')
    if not manifest:
        raise ValueError('Missing artifact manifest')
    expected = {row['path']: row for row in manifest['artifacts']}
    actual_paths = {str(path.relative_to(directory)) for path in directory.rglob('*')
                    if path.is_file() and path.name != 'manifest.json'}
    if set(expected) != actual_paths:
        raise ValueError('Manifest file inventory mismatch')
    for relative, row in expected.items():
        path = directory / relative
        if (path.stat().st_size != row['size_bytes'] or
                hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']):
            raise ValueError('Manifest hash/size mismatch: ' + relative)
    return dict(status='passed', files=len(expected), hashes_verified=len(expected),
                inventory_exact=True, algorithm='sha256')
