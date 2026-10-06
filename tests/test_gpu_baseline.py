"""Orchestration tests use a scripted policy; never import torch or load weights."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_gpu_baseline as baseline
from test_policy_rollouts import ScriptPolicy, action


class BaselineTests(unittest.TestCase):
    def test_eight_original_questions_and_artifacts(self):
        public = json.loads((ROOT / 'data/tasks.public.json').read_text())
        evaluation = json.loads((ROOT / 'data/tasks.evaluation.json').read_text())
        selected = baseline.select_tasks(public, evaluation, 8)
        policy = ScriptPolicy([action('lookup', entity='s99', field='advisor'),
                               action('direct_answer', final_answer='unknown')])
        policy.runtime_metadata = {'device': 'cuda:0', 'dtype': 'bfloat16'}
        with tempfile.TemporaryDirectory() as temp, patch.object(baseline, 'HuggingFacePolicy', return_value=policy) as factory:
            output = Path(temp) / 'baseline'
            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                summary = baseline.run_baseline(output, Path(temp) / 'cache')
            config = factory.call_args.args[0]
            original = json.loads((ROOT / 'configs/policy.json').read_text())['policy']
            self.assertEqual(config['model_name'], original['model_name'])
            self.assertEqual(config['revision'], original['revision'])
            self.assertEqual(config['device'], 'cuda:0')
            self.assertEqual(config['dtype'], 'auto')
            for flag in ('cpu_fallback', 'do_sample'):
                self.assertFalse(config[flag])
            self.assertTrue(config['local_files_only'])
            self.assertEqual(config['cache_dir'], str((Path(temp) / 'cache').resolve()))
            self.assertEqual(len(policy.sessions), 8)
            self.assertEqual([s.states[0].question for s in policy.sessions], [t['question'] for t in selected])
            self.assertTrue(all(s.states[0].history == [] for s in policy.sessions))
            ids = [t['task_id'] for t in selected]
            self.assertEqual(summary['run']['task_ids'], ids)
            self.assertEqual(json.loads((output / 'selected_task_ids.json').read_text()), ids)
            records = [json.loads(line) for line in (output / 'policy_rollouts.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 8)
            for record in records:
                self.assertEqual(record['metadata']['rollout_id'], 0)
                self.assertEqual(record['metadata']['rollout_config'], dict(max_steps=4, max_tool_calls=3, max_parse_retries=1, seed=2026))
            for row in summary['task_level']:
                self.assertTrue({'correctness', 'total_cost', 'number_of_tool_calls', 'tool_sequence',
                                 'deep_search_used', 'direct_answer_without_tool', 'invalid_action_attempts',
                                 'invalid_arguments', 'repair_attempts', 'tool_execution_errors',
                                 'termination_reason', 'truncated'} <= row.keys())
                self.assertEqual(row['tool_execution_errors'], 1)
                self.assertNotIn('cost_aware_reward', row)
            rewards = [json.loads(line) for line in (output / 'mock_ai_rewards.jsonl').read_text().splitlines()]
            self.assertEqual(len(rewards), 8)
            self.assertTrue(all(r['scorer'] == 'CostAwareReward(MockAIReward)' for r in rewards))
            for name in ('experiment_config.json', 'runtime_metadata.json', 'policy_rollout_summary.json'):
                self.assertTrue(json.loads((output / name).read_text()))
            self.assertIn('MockAIReward', (output / 'report.md').read_text())
            self.assertIn('Normal completions: 8', terminal.getvalue())
            self.assertIn('Tool errors: 8', terminal.getvalue())

    def test_existing_directory_rejected_before_policy_load(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(baseline, 'HuggingFacePolicy') as factory:
            sentinel = Path(temp) / 'policy_rollouts.jsonl'
            sentinel.write_text('existing experiment\n')
            with self.assertRaises(FileExistsError):
                baseline.run_baseline(temp, Path(temp) / 'cache')
            factory.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'existing experiment\n')


if __name__ == '__main__':
    unittest.main()
