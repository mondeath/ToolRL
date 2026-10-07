"""Frozen data, real environment rollout, masks, reward and preflight tests."""
from copy import copy, deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from src.env_v1 import generate_kb, eligibility
from src.env_v1.tasks import generate_tasks, initial_state
from src.env_v1.evaluation import evaluate
from src.policies import PolicyOutput
from src.policies.schema import build_messages
from src.training.grpo_v1.data import load_data, read_json, validate_config
from src.training.grpo_v1.loss import make_segment, validate_segment, group_advantages, group_rates
from src.training.grpo_v1.collector import collect_group, task_reward
from src.training.grpo_v1.preflight import preflight, check_snapshot
from src.training.grpo_v1.evaluation import greedy_mode

CONFIG = read_json(ROOT / 'configs/grpo_v1_smoke.json')


class ScriptedCapturePolicy:
    """Only a fixture: no private task/evaluator state goes to the collector."""
    model_name = 'scripted-fixture'
    runtime_metadata = {}

    def __init__(self, plans):
        self.plans = plans
        self.config = dict(CONFIG['sampling'])
        self.sampling_config = dict(CONFIG['sampling'])
        self.count = 0
        self.capture = True
        self.sessions = []

    def new_session(self, seed):
        session = copy(self)
        session.seed, session.index, session.segments = seed, 0, []
        session.plan = self.plans[self.count]
        self.count += 1
        self.last_session = session
        self.sessions.append(session)
        return session

    def act(self, state):
        messages = build_messages(state)
        prompt_text = json.dumps(messages)
        raw = self.plan[self.index]
        self.index += 1
        tokens = [ord(char) for char in raw] + [0]  # EOS stays a target.
        segment = make_segment(messages, [ord(c) for c in prompt_text], tokens, raw,
                               old_logprobs=[-0.5] * len(tokens))
        self.segments.append(segment)
        return PolicyOutput(raw, tokens)


def action(name, **arguments):
    return json.dumps(dict(action=name, arguments=arguments))


class GRPOCoreTests(unittest.TestCase):
    def test_fixed_selection_balanced_multiseed_and_disjoint(self):
        train, diagnostic = load_data(ROOT, CONFIG)
        self.assertEqual(len(train), 24)
        self.assertEqual({x['seed'] for x in train}, set(range(1000, 1008)))
        self.assertEqual([sum(x['task']['difficulty'] == level for x in train) for level in ('L1', 'L2', 'L3')], [8, 8, 8])
        self.assertEqual([x['task']['task_id'] for x in diagnostic], read_json(ROOT / 'configs/env_v1_diagnostic_tasks.json')['task_ids'])
        self.assertFalse({x['task']['task_id'] for x in train} & {x['task']['task_id'] for x in diagnostic})
        bad = deepcopy(CONFIG)
        bad['generator_source_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'generator'):
            load_data(ROOT, bad)
        bad = deepcopy(CONFIG)
        bad['tasks_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'task list'):
            load_data(ROOT, bad)

    def test_dev_and_diagnostic_injected_selection_rejected(self):
        from src.evaluation.model_scale import digest
        import src.training.grpo_v1.data as data
        selection = read_json(ROOT / CONFIG['tasks_file'])
        for split, seed in [('diagnostic', 42), ('dev', 2000)]:
            changed = deepcopy(selection)
            changed['tasks'][0].update(split=split, seed=seed)
            bad = dict(CONFIG, tasks_sha256=digest(changed))
            original = data.read_json
            def reader(path):
                return changed if str(path).endswith('grpo_v1_smoke_tasks.json') else original(path)
            with patch.object(data, 'read_json', side_effect=reader):
                with self.assertRaisesRegex(ValueError, 'non-train'):
                    load_data(ROOT, bad)

    def test_groups_reward_and_observation_mask(self):
        kb = generate_kb()
        task = generate_tasks(kb)[0]
        plans = [[action('lookup', entity='s00'), action('direct_answer', final_answer=value)]
                 for value in (3.8, 0, 3.8, 0)]
        policy = ScriptedCapturePolicy(plans)
        group = collect_group(dict(kb=kb, task=task, seed=42), policy, CONFIG, 'test-group')
        self.assertEqual([r['reward'] for r in group], [1, 0, 1, 0])
        self.assertEqual({r['task_id'] for r in group}, {task['task_id']})
        self.assertEqual(len({r['session_seed'] for r in group}), 4)
        self.assertEqual(len({id(s.segments) for s in policy.sessions}), 4)
        for row in group:
            self.assertEqual(row['termination_reason'], 'direct_answer')
            self.assertEqual(len(row['segments']), 2)
            first, second = row['segments']
            self.assertIn('observation', second['messages'][1]['content'])
            self.assertIn('3.8', second['messages'][1]['content'])
            # Character tokenizer makes an actual observation location testable.
            text = ''.join(map(chr, second['prompt_token_ids']))
            start = text.index('3.8')
            self.assertEqual(second['generated_token_mask'][start:start + 3], [0, 0, 0])
            self.assertEqual(sum(second['generated_token_mask'][:len(second['prompt_token_ids'])]), 0)
            self.assertEqual(sum(second['generated_token_mask']), len(second['generated_token_ids']))
            self.assertEqual(second['generated_token_ids'][-1], 0)
            self.assertEqual(second['generated_token_mask'][-1], 1)
        group[0]['segments'][0]['messages'][0]['content'] = 'mutation'
        self.assertNotEqual(group[1]['segments'][0]['messages'][0]['content'], 'mutation')

    def test_reward_is_only_evaluator_task_success(self):
        self.assertEqual(task_reward(dict(task_success=True, success=False, tool_cost=999,
                                         rule_violations=['duplicate_submit'], trajectory_length=999)), 1)
        self.assertEqual(task_reward(dict(task_success=False, tool_cost=0, rule_violations=[])), 0)
        kb = generate_kb()
        task = next(t for t in generate_tasks(kb) if t['difficulty'] == 'L3')
        state = initial_state(kb, task)
        sc = task['evaluation']['scholarship_id']
        eligible = [sid for sid in state['students'] if eligibility(state, sid, sc)['eligible']]
        # Duplicate attempt produces a violation but evaluator success remains 1.
        plan = [action('submit', student_ids=eligible, scholarship_id=sc), action('direct_answer', final_answer='done')]
        policy = ScriptedCapturePolicy([plan] * 4)
        group = collect_group(dict(kb=kb, task=task, seed=42), policy, CONFIG, 'state-reset')
        self.assertEqual([r['reward'] for r in group], [1] * 4)
        self.assertTrue(all(len(r['rule_violations']) == 1 for r in group))
        self.assertTrue(all(r['observations'] == group[0]['observations'] for r in group))
        self.assertEqual(kb, generate_kb())

    def test_parse_repair_and_invalid_actions_still_have_tokens(self):
        kb = generate_kb()
        task = generate_tasks(kb)[0]
        plans = [['not json', action('direct_answer', final_answer=3.8)], ['bad', 'bad'],
                 ['bad', 'bad'], ['not json', action('direct_answer', final_answer=3.8)]]
        group = collect_group(dict(kb=kb, task=task, seed=42), ScriptedCapturePolicy(plans), CONFIG, 'repair')
        self.assertEqual([r['reward'] for r in group], [1, 0, 0, 1])
        self.assertTrue(all(len(r['segments']) == 2 for r in group))
        self.assertIn('format_feedback', group[0]['segments'][1]['messages'][1]['content'])
        self.assertEqual(group[1]['termination_reason'], 'invalid_action')
        altered = deepcopy(group[0]['segments'][1])
        altered['generated_token_mask'][1] = 1
        with self.assertRaisesRegex(ValueError, 'mask'):
            validate_segment(altered)

    def test_budget_blocked_generation_captured_and_runtime_error_not_reward(self):
        kb = generate_kb()
        item = dict(kb=kb, task=generate_tasks(kb)[0], seed=42)
        plans = [[action('lookup', entity='s00')] * 9 for _ in range(4)]
        group = collect_group(item, ScriptedCapturePolicy(plans), CONFIG, 'budget')
        assert all(r['termination_reason'] == 'max_tool_calls' and len(r['segments']) == 9 for r in group)
        assert all(len(r['observations']) == 8 and r['reward'] == 0 for r in group)
        with patch.object(ScriptedCapturePolicy, 'act', side_effect=RuntimeError('synthetic infrastructure failure')):
            with self.assertRaisesRegex(RuntimeError, 'infrastructure failure'):
                collect_group(item, ScriptedCapturePolicy(plans), CONFIG, 'failed')

    def test_all_equal_advantages_and_group_rates(self):
        for rewards in ([1] * 4, [0] * 4):
            self.assertEqual(group_advantages(rewards), [0] * 4)
        self.assertAlmostEqual(sum(group_advantages([1, 0, 1, 0])), 0)
        rates = group_rates([[1] * 4, [0] * 4, [1, 0, 1, 0]])
        self.assertEqual(list(rates.values()), [1 / 3] * 3)
        for bad in ([], [float('nan')], [0.5]):
            with self.assertRaises(ValueError):
                group_advantages(bad)

    def test_greedy_mode_restores_sampling_even_on_exception(self):
        policy = ScriptedCapturePolicy([])
        with self.assertRaises(RuntimeError):
            with greedy_mode(policy):
                self.assertFalse(policy.config['do_sample'])
                self.assertFalse(policy.capture)
                raise RuntimeError('test')
        self.assertTrue(policy.config['do_sample'])
        self.assertTrue(policy.capture)

    def test_static_preflight_does_not_load_model(self):
        with patch('src.training.grpo_v1.policy.load_policy', side_effect=AssertionError('must not load')):
            report = preflight(ROOT, CONFIG, require_gpu=False)
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['training_started'])
        with self.assertRaises(ValueError):
            check_snapshot(None, CONFIG)
        with self.assertRaises(ValueError):
            check_snapshot('/tmp/arbitrary-model', CONFIG)
        bad = deepcopy(CONFIG)
        bad['training']['max_updates'] = 10
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_gpu_preflight_metadata_path_without_loading_or_gpu_compute(self):
        import importlib
        import importlib.metadata
        import torch
        original_import = importlib.import_module
        original_version = importlib.metadata.version
        def fake_import(name, *args, **kwargs):
            return SimpleNamespace() if name in ('bitsandbytes', 'peft') else original_import(name, *args, **kwargs)
        def fake_version(name):
            return {'bitsandbytes': '0.48.1', 'peft': '0.17.1'}.get(name) or original_version(name)
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / 'models--Qwen--Qwen2.5-7B-Instruct' / 'snapshots' / CONFIG['revision']
            snapshot.mkdir(parents=True)
            (snapshot / 'config.json').write_text(json.dumps(dict(model_type='qwen2', hidden_size=3584,
                num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4, vocab_size=152064)))
            for file in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'model.safetensors'):
                (snapshot / file).write_text('fixture: readable only, not real weights')
            with patch('importlib.import_module', side_effect=fake_import), \
                 patch('importlib.metadata.version', side_effect=fake_version), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'is_bf16_supported', return_value=True), \
                 patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(name='FAKE GPU', total_memory=24 * 1024 ** 3)), \
                 patch('src.training.grpo_v1.policy.load_policy', side_effect=AssertionError('must not load')):
                result = preflight(ROOT, CONFIG, snapshot)
            self.assertEqual(result['status'], 'passed')
            self.assertFalse(result['model_loaded'])
            self.assertFalse(result['training_started'])
            self.assertEqual(result['checks']['gpu']['name'], 'FAKE GPU')

    def test_archived_bf16_results_are_separate(self):
        from run_grpo_v1_smoke import archived_baseline
        _, diagnostic = load_data(ROOT, CONFIG)
        archive = archived_baseline(ROOT, CONFIG, diagnostic)
        self.assertEqual(archive['metrics']['task_count'], 15)
        self.assertAlmostEqual(archive['metrics']['overall_success'], 11 / 15)


if __name__ == '__main__':
    unittest.main()
