"""Baseline data, one-pass schedule, accounting and reporting without a GPU."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from src.training.grpo_v1.data import read_json
from src.training.grpo_v1.baseline_data import load_baseline_data, validate_config, dry_run, preflight_data
from src.training.grpo_v1.baseline_runner import schedule, validate_checkpoint, write_json
from src.training.grpo_v1.baseline_reporting import group_metrics, training_summary, render_report
from src.training.grpo_v1.preflight import preflight

CONFIG = read_json(ROOT / 'configs/grpo_v1_baseline.json')


def test_fixed_240_balanced_spread_and_isolation():
    train, dev, diagnostic = load_baseline_data(ROOT, CONFIG)
    assert len(train) == 240 and len(dev) == 400 and len(diagnostic) == 15
    assert Counter(x['seed'] for x in train) == {seed: 8 for seed in range(1000, 1030)}
    assert Counter(x['task']['difficulty'] for x in train) == {'L1': 80, 'L2': 80, 'L3': 80}
    for start in range(0, 240, 60):
        assert len({x['seed'] for x in train[start:start + 60]}) == 30
    assert not {x['task']['task_id'] for x in train} & {x['task']['task_id'] for x in dev + diagnostic}
    assert [x['task']['task_id'] for x in train] == [x['task_id'] for x in read_json(ROOT / CONFIG['tasks_file'])['tasks']]


@pytest.mark.parametrize('field,value', [('learning_rate', 2e-5), ('epochs', 2), ('max_updates', 3),
                                         ('weight_decay', 0.01), ('max_grad_norm', 2), ('beta', .01)])
def test_no_hyperparameter_changes(field, value):
    modified = deepcopy(CONFIG)
    modified['training'][field] = value
    with pytest.raises(ValueError):
        validate_config(ROOT, modified)


def test_selection_mutation_and_eval_seed_injection_rejected():
    import src.training.grpo_v1.baseline_data as module
    from src.evaluation.model_scale import digest
    modified = deepcopy(CONFIG)
    modified['tasks_sha256'] = 'changed'
    with pytest.raises(ValueError, match='selection hash'):
        load_baseline_data(ROOT, modified)
    selection = read_json(ROOT / CONFIG['tasks_file'])
    selection['tasks'][0]['split'] = 'dev'
    modified['tasks_sha256'] = digest(selection)
    original = module.read_json
    with patch.object(module, 'read_json', side_effect=lambda p: selection if Path(p).name == 'grpo_v1_baseline_tasks.json' else original(p)):
        with pytest.raises(ValueError, match='Non-train'):
            load_baseline_data(ROOT, modified)


def test_dry_run_counts_and_preflight_do_not_load_model():
    with patch('src.training.grpo_v1.policy.load_policy', side_effect=AssertionError('no model')):
        result = dry_run(ROOT, CONFIG)
        check = preflight(ROOT, CONFIG, require_gpu=False, data_loader=preflight_data)
    assert result['attempted_groups'] == result['maximum_optimizer_steps'] == 240
    assert result['training_rollouts'] == 960 and result['evaluation_rollouts'] == 2015
    assert result['total_rollouts'] == 2975 and result['maximum_training_generation_calls'] == 17280
    assert not result['training_started'] and check['status'] == 'passed'
    assert result['estimate']['kind'].startswith('uncalibrated')


def test_single_pass_checkpoints_dev_curve_final_only_diagnostic():
    train, dev, diagnostic = load_baseline_data(ROOT, CONFIG)
    calls, collected, logged = [], [], []
    steps = 0
    def collect(item, attempted):
        collected.append(item['task']['task_id'])
        return attempted
    def update(group):
        nonlocal steps
        mixed = group % 3 == 0
        steps += mixed
        return dict(update_applied=mixed)
    def save(attempted):
        calls.append(('save', attempted, steps))
    def evaluate(items, attempted, split):
        calls.append((split, attempted, len(items)))
    schedule(train, dev, diagnostic, CONFIG, collect, update, save, evaluate,
             lambda group, result, attempted: logged.append((attempted, result)))
    assert len(set(collected)) == len(collected) == len(logged) == 240
    assert steps == 80  # Goes past smoke's three-step cap; equal groups never re-sampled.
    assert [x[1] for x in calls if x[0] == 'save'] == [0, 60, 120, 180, 240]
    assert [x[1] for x in calls if x[0] == 'dev'] == [0, 60, 120, 180, 240]
    assert [x for x in calls if x[0] == 'diagnostic'] == [('diagnostic', 240, 15)]
    assert calls[0] == ('save', 0, 0) and calls[1] == ('dev', 0, 400)
    assert calls[-1] == ('diagnostic', 240, 15)


def test_error_aborts_no_retry_or_adjustment():
    calls = []
    def collect(item, attempted):
        calls.append(attempted)
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='out of memory'):
        schedule([1] * 240, [], [], CONFIG, collect, lambda g: None,
                 lambda n: None, lambda *args: None, lambda *args: None)
    assert calls == [1]


def test_diagnostic_checkpoint_gate_before_model_load(tmp_path):
    checkpoint = tmp_path / 'adapter'
    checkpoint.mkdir()
    metadata = dict(config=CONFIG, selection_sha256=CONFIG['tasks_sha256'], attempted_groups=60)
    write_json(checkpoint / 'metadata.json', metadata)
    assert validate_checkpoint(ROOT, CONFIG, checkpoint, 'dev')['attempted_groups'] == 60
    with pytest.raises(ValueError, match='final checkpoint'):
        validate_checkpoint(ROOT, CONFIG, checkpoint, 'diagnostic')
    metadata['attempted_groups'] = 240
    write_json(checkpoint / 'metadata.json', metadata)
    assert validate_checkpoint(ROOT, CONFIG, checkpoint, 'diagnostic')['attempted_groups'] == 240


def make_group(rewards, level='L1'):
    call = dict(tool_name='lookup', arguments=dict(entity='s00'), cost=1)
    return [dict(task_id='test', difficulty=level, seed=1000, reward=r, success=bool(r),
        trajectory_length=3, termination_reason='direct_answer' if r else 'invalid_action',
        tool_cost=2, rule_violations=[], generated_token_count=20,
        trajectory=dict(tool_calls=[call, call], termination_reason='direct_answer')) for r in rewards]


def test_metrics_accounting_and_reward_distribution():
    groups = []
    for i, rewards in enumerate([[1, 0, 1, 0], [1] * 4, [0] * 4], 1):
        update = dict(step=1, advantages=[0] * 4, update_applied=i == 1, policy_loss=0., grad_norm=1.,
                      learning_rate=1e-5, adapter_delta_l2=.01 if i == 1 else 0., kl=None)
        groups.append(group_metrics(make_group(rewards), update, i, 10. * i, dict(gpu_peak_memory=100)))
    summary = training_summary(groups)
    assert summary['attempted_groups'] == 3 and summary['optimizer_steps'] == 1
    assert summary['mixed_reward_group_rate'] == 1 / 3
    assert summary['all_success_group_count'] == summary['all_failure_group_count'] == 1
    assert summary['reward_distribution'] == {'1': 6, '0': 6}
    assert summary['cumulative_task_success'] == .5
    assert summary['success_by_difficulty']['L2']['success_rate'] is None
    assert groups[0]['repeated_actions'] == [1] * 4
    assert groups[0]['generated_token_count'] == 80
    assert groups[0]['invalid_action_rate'] == .5


def test_reporting_has_required_metrics_and_does_not_invent_results(tmp_path):
    write_json(tmp_path / 'dry_run.json', dry_run(ROOT, CONFIG))
    path = render_report(tmp_path)
    content = path.read_text()
    assert 'avg cost successful' in content and 'deep_search usage' in content and 'Dev curve' in content
    assert '不能回答' in content and '| overall success | — | — |' in content
    for folder in ('base_dev', 'eval_group_240/dev'):
        (tmp_path / folder).mkdir(parents=True)
    write_json(tmp_path / 'base_dev/summary.json', dict(overall_success=.8))
    write_json(tmp_path / 'eval_group_240/dev/summary.json', dict(overall_success=.7))
    write_json(tmp_path / 'status.json', dict(status='completed'))
    assert '下降' in render_report(tmp_path).read_text()


def test_real_lora_updates_continue_past_three():
    pytest.importorskip('peft')
    import torch
    from src.training.grpo_v1.synthetic import tiny_lora_model, synthetic_group
    from src.training.grpo_v1.trainer import GRPOTrainer, base_digest
    torch.set_num_threads(2)
    model = tiny_lora_model()
    base = base_digest(model)
    trainer = GRPOTrainer(model, CONFIG['training'])
    for _ in range(5):
        result = trainer.update(synthetic_group(model))
        assert result['update_applied'] and result['adapter_delta_l2'] > 0
    assert trainer.step == 5 and base == base_digest(model)
    group = synthetic_group(model)
    for row in group:
        row['reward'] = 1
    assert not trainer.update(group)['update_applied'] and trainer.step == 5


def test_full_runner_wiring_with_tiny_cpu_and_synthetic_rollouts(tmp_path):
    """Exercise files/checkpoints/report over 240 groups; no real task inference."""
    pytest.importorskip('peft')
    import torch
    from types import SimpleNamespace
    from src.training.grpo_v1.synthetic import tiny_lora_model, synthetic_group, SyntheticTokenizer
    from src.training.grpo_v1.policy import parameter_counts
    from src.training.grpo_v1.baseline_runner import train_baseline
    torch.set_num_threads(2)
    model = tiny_lora_model()
    policy = SimpleNamespace(model=model, tokenizer=SyntheticTokenizer(), runtime_metadata=parameter_counts(model))
    collected, evaluations = [], []
    def fake_collect(item, policy, config, group_id):
        group = synthetic_group(model)
        collected.append(item['task']['task_id'])
        mixed = len(collected) in (1, 240)
        for row in group:
            row.update(task_id=item['task']['task_id'], difficulty=item['task']['difficulty'], seed=item['seed'],
                trajectory_length=2, tool_cost=1, rule_violations=[], generated_token_count=5,
                termination_reason='direct_answer', trajectory=dict(tool_calls=[], termination_reason='direct_answer'))
            if not mixed:
                row['reward'] = 1.
            row['success'] = bool(row['reward'])
        return group
    def fake_eval(items, policy, config, directory):
        evaluations.append(str(directory.relative_to(tmp_path / 'run')))
        directory.mkdir(parents=True)
        result = dict(overall_success=.5, task_count=len(items), success_by_difficulty=dict(L1=.5, L2=.5, L3=.5))
        write_json(directory / 'summary.json', result)
        return result
    with patch('src.training.grpo_v1.policy.load_policy', return_value=policy), \
         patch('src.training.grpo_v1.collector.collect_group', side_effect=fake_collect), \
         patch('src.training.grpo_v1.evaluation.greedy_eval', side_effect=fake_eval), \
         patch.object(torch.cuda, 'reset_peak_memory_stats'):
        status = train_baseline(ROOT, CONFIG, '/synthetic-only', tmp_path / 'run', dict(synthetic=True))
    assert status['status'] == 'completed' and status['base_unchanged'] and status['adapter_changed']
    assert status['optimizer_steps'] == 2 and len(set(collected)) == 240
    assert evaluations == ['base_dev', 'eval_group_060/dev', 'eval_group_120/dev',
                           'eval_group_180/dev', 'eval_group_240/dev', 'eval_group_240/diagnostic']
    assert len((tmp_path / 'run/rollouts.jsonl').read_text().splitlines()) == 960
    summary = read_json(tmp_path / 'run/training_summary.json')
    assert summary['mixed_group_count'] == summary['optimizer_steps'] == 2
    for attempted in CONFIG['checkpoint_groups']:
        assert (tmp_path / f'run/checkpoints/group_{attempted:03}/adapter_model.safetensors').is_file()
        assert not (tmp_path / f'run/checkpoints/group_{attempted:03}/model.safetensors').exists()
    assert '不变' in (tmp_path / 'run/GRPO_V1_BASELINE_RESULTS.md').read_text()
