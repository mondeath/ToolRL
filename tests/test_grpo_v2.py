"""CPU-only correctness checks for the aligned V2 experiment contract."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.grpo_v1.loss import ratio_diagnostics
from src.training.grpo_v1.baseline_reporting import training_summary
from src.training.grpo_v2.alignment import validate_sampling_logprob_alignment
from src.training.grpo_v2.data import (validate_config, dry_run, validate_smoke_config,
    smoke_dry_run, load_v2_smoke_data)
from src.training.grpo_v2.holdout import generate_tasks, generate_world, load_frozen_holdout, outcome
from src.training.grpo_v2.reporting import write_manifest, validate_artifact_manifest
from src.training.grpo_v2.runner import audit_training_outputs

CONFIG = json.loads((ROOT / 'configs/grpo_v2_baseline.json').read_text())
SMOKE_CONFIG = json.loads((ROOT / 'configs/grpo_v2_gpu_smoke.json').read_text())
HOLDOUT_CONFIG = json.loads((ROOT / 'configs/grpo_v2_holdout.json').read_text())


def test_v2_config_is_independent_and_aligned():
    alignment = validate_config(ROOT, CONFIG)
    assert alignment['status'] == 'aligned'
    assert CONFIG['version'] == 2 and CONFIG['output_namespace'] == 'results/grpo_v2'
    plan = dry_run(ROOT, CONFIG)
    assert plan['training_rollouts'] == 960 and plan['passes'] == 1
    assert 'largely inactive' in plan['limitation']


@pytest.mark.parametrize('field,value', [
    ('do_sample', False), ('temperature', 0.8), ('top_p', 0.95), ('top_k', 20),
])
def test_incompatible_behavior_policy_is_rejected(field, value):
    sampling = deepcopy(CONFIG['sampling'])
    sampling[field] = value
    with pytest.raises(ValueError, match='Sampling/log-prob space mismatch'):
        validate_sampling_logprob_alignment(sampling, 'raw_model_softmax')
    bad = deepcopy(CONFIG)
    bad['sampling'][field] = value
    with pytest.raises(ValueError, match='Sampling/log-prob space mismatch'):
        validate_config(ROOT, bad)


def test_ratio_clip_fraction_and_approx_kl_math():
    ratios = torch.tensor([1.0, 1.5, 0.5])
    diagnostic = ratio_diagnostics(ratios.log(), torch.zeros(3), epsilon=0.2)
    assert diagnostic['token_count'] == 3
    assert diagnostic['ratio_sum'] == pytest.approx(3.0)
    assert diagnostic['ratio_min'] == pytest.approx(0.5)
    assert diagnostic['ratio_max'] == pytest.approx(1.5)
    assert diagnostic['clipped_token_count'] == 2
    expected_kl = sum(float((r - 1) - torch.log(r)) for r in ratios)
    assert diagnostic['approx_kl_sum'] == pytest.approx(expected_kl)


def test_reward_variance_zero_groups_and_weighted_diagnostics_summary():
    common = dict(difficulty='L1', rewards=[1, 0, 1, 0], generated_token_count=10,
                  gpu_peak_memory=None, optimizer_step_performed=True,
                  mixed_reward_group=True, all_success_group=False, all_failure_group=False)
    groups = [dict(common, reward_variance=.25, zero_variance_group=False,
                   ratio_mean=1.0, clip_fraction=0.0, approx_kl=0.0, ratio_token_count=10),
              dict(common, rewards=[1, 1, 1, 1], optimizer_step_performed=False,
                   mixed_reward_group=False, all_success_group=True,
                   reward_variance=0.0, zero_variance_group=True,
                   ratio_mean=None, clip_fraction=0.0, approx_kl=0.0, ratio_token_count=0)]
    result = training_summary(groups)
    assert result['optimizer_steps'] == 1
    assert result['zero_variance_groups'] == 1
    assert result['mean_reward_variance'] == pytest.approx(.125)
    assert result['ratio_mean'] == pytest.approx(1.0)
    assert result['clip_fraction'] == 0.0 and result['approx_kl'] == 0.0


def test_v2_manifest_records_seeds_rollouts_checkpoints_and_raw_eval(tmp_path):
    (tmp_path / 'status.json').write_text('{"status":"completed","optimizer_steps":1}\n')
    (tmp_path / 'config.json').write_text('{}\n')
    (tmp_path / 'tasks.json').write_text('{}\n')
    (tmp_path / 'rollouts.jsonl').write_text('{"task_id":"x"}\n')
    (tmp_path / 'metrics.jsonl').write_text('{"ratio_mean":1.0}\n')
    checkpoint = tmp_path / 'checkpoints' / 'group_000'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'metadata.json').write_text('{}\n')
    manifest = write_manifest(tmp_path, CONFIG, [dict(seed=1000), dict(seed=1001)])
    assert manifest['random_seeds']['global_seed'] == 2026
    assert manifest['random_seeds']['train_world_seeds'] == [1000, 1001]
    required = manifest['required_artifacts']
    assert required['training_rollouts'] == 'rollouts.jsonl'
    assert required['adapters_and_checkpoints'] == 'checkpoints/'
    assert 'base_dev/results.jsonl' in required['raw_evaluation_trajectories']
    assert {row['path'] for row in manifest['artifacts']} >= {
        'config.json', 'tasks.json', 'rollouts.jsonl', 'metrics.jsonl',
        'checkpoints/group_000/metadata.json', 'status.json'}
    assert validate_artifact_manifest(tmp_path)['hashes_verified'] == len(manifest['artifacts'])
    (tmp_path / 'metrics.jsonl').write_text('tampered\n')
    with pytest.raises(ValueError, match='hash/size mismatch'):
        validate_artifact_manifest(tmp_path)


def test_v2_adapter_checkpoint_reload_preserves_logprobs(tmp_path):
    pytest.importorskip('peft', reason='Install requirements-training.txt for adapter reload test')
    from types import SimpleNamespace
    from src.training.grpo_v1.loss import generated_logprobs
    from src.training.grpo_v1.policy import assert_adapter_only
    from src.training.grpo_v1.synthetic import tiny_lora_model, synthetic_group, SyntheticTokenizer
    from src.training.grpo_v1.trainer import GRPOTrainer, save_checkpoint, reload_adapter_in_place
    model = tiny_lora_model()
    group = synthetic_group(model)
    trainer = GRPOTrainer(model, CONFIG['training'])
    trainer.update(group)
    policy = SimpleNamespace(model=model, tokenizer=SyntheticTokenizer())
    expected = generated_logprobs(model, group[0]['segments'][0]).detach()
    checkpoint = tmp_path / 'adapter'
    save_checkpoint(policy, trainer, checkpoint, dict(version=2))
    with torch.no_grad():
        for _, parameter in assert_adapter_only(model):
            parameter.add_(0.5)
    assert not torch.allclose(generated_logprobs(model, group[0]['segments'][0]), expected)
    reload_adapter_in_place(policy, checkpoint)
    torch.testing.assert_close(generated_logprobs(model, group[0]['segments'][0]), expected)


def test_gpu_smoke_is_bounded_aligned_and_excludes_holdout():
    assert validate_smoke_config(ROOT, SMOKE_CONFIG)['status'] == 'aligned'
    train, dev, diagnostic = load_v2_smoke_data(ROOT, SMOKE_CONFIG)
    assert len(train) == 24 and len(dev) == len(diagnostic) == 15
    plan = smoke_dry_run(ROOT, SMOKE_CONFIG)
    assert plan['training_rollouts'] == 96 and plan['maximum_optimizer_steps'] == 24
    assert plan['checkpoint_attempted_groups'] == [0, 24]
    assert not plan['holdout_evaluation_enabled'] and plan['smoke_only']


def test_frozen_holdout_hashes_determinism_balance_and_overlap_audit():
    items, manifest = load_frozen_holdout(ROOT, HOLDOUT_CONFIG)
    assert len(items) == 72
    assert manifest['tasks_by_difficulty'] == {'L1': 24, 'L2': 24, 'L3': 24}
    audit = manifest['overlap_audit']
    assert audit['exact_instance_overlap'] == 0
    assert audit['question_overlap'] == 0
    assert audit['template_overlap'] == 0
    assert audit['entity_overlap'] == 0
    assert audit['answer_overlap'] == 12
    assert manifest['usage_policy'].startswith('Hold-out is report-only')
    for seed in HOLDOUT_CONFIG['seeds']:
        persisted = [item for item in items if item['seed'] == seed]
        kb = generate_world(seed)
        assert persisted[0]['kb'] == kb
        assert [item['task'] for item in persisted] == generate_tasks(kb, seed)


def test_holdout_expected_outcomes_and_state_targets_are_valid():
    from src.env_v1 import eligibility
    from src.env_v1.evaluation import evaluate
    from src.env_v1 import Environment
    from src.env_v1.tasks import initial_state
    from src.tasks import public_task
    items, _ = load_frozen_holdout(ROOT, HOLDOUT_CONFIG)
    for item in items:
        task, kb = item['task'], item['kb']
        if task['difficulty'] != 'L3':
            env = Environment(initial_state(kb, task), failure_rate=0)
            env.reset(public_task(task))
            env.step('direct_answer', final_answer=outcome(kb, task))
            assert evaluate(task, env)['task_success']
        else:
            initial = task['initial_applications']
            target = task['final_assertions'][0]['expected']
            assert all(target[key] == value for key, value in initial.items())
            for key, application in target.items():
                if key not in initial:
                    assert eligibility(initial_state(kb, task), application['student_id'],
                                       application['scholarship_id'])['eligible']


def test_post_training_audit_checks_group_updates_and_output_completeness(tmp_path):
    config = dict(SMOKE_CONFIG, task_count=2, group_size=4)
    mixed = dict(mixed_reward_group=True, zero_variance_group=False,
        optimizer_step_performed=True, optimizer_steps=1, reward_variance=.25,
        adapter_delta_l2=.01, ratio_token_count=8, ratio_min=1.0, ratio_mean=1.0,
        ratio_max=1.0, clip_fraction=0.0, approx_kl=0.0)
    equal = dict(mixed_reward_group=False, zero_variance_group=True,
        optimizer_step_performed=False, optimizer_steps=1, reward_variance=0.0,
        adapter_delta_l2=0.0, ratio_token_count=0, ratio_min=None, ratio_mean=None,
        ratio_max=None, clip_fraction=0.0, approx_kl=0.0)
    (tmp_path / 'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in (mixed, equal)) + '\n')
    rollout = dict(segments=[dict(old_logprobs=[-.1, -.2], generated_token_ids=[1, 2])])
    (tmp_path / 'rollouts.jsonl').write_text((json.dumps(rollout) + '\n') * 8)
    for relative in ('config.json', 'tasks.json', 'training_summary.json', 'status.json',
                     'base_dev/results.jsonl', 'eval_group_002/dev/results.jsonl',
                     'eval_group_002/diagnostic/results.jsonl',
                     'checkpoints/group_002/training_state.pt'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    (tmp_path / 'checkpoint_group_002.json').write_text(json.dumps(
        dict(checkpoint_reload=True, max_reload_logprob_error=0.0)) + '\n')
    audit = audit_training_outputs(tmp_path, config)
    assert audit['optimizer_steps'] == audit['mixed_reward_groups'] == 1
    assert audit['zero_variance_groups'] == 1 and audit['checkpoint_reload']
