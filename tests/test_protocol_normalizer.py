import json
import unittest
from unittest.mock import patch

from test_policy_rollouts import ScriptPolicy, action
from src.environment import ToolEnvironment
from src.kb import generate_kb
from src.policies.normalizer import normalize_protocol_output
from src.policies.parser import parse_action
from src.rollouts import rollout, RolloutConfig


class NormalizerTests(unittest.TestCase):
    def test_allowed_prefixes_and_exact_payload_preservation(self):
        cases = [
            ('lookup', '{"entity":"s00","field":"advisor"}'),
            ('search', '{"query":"Student"}'),
            ('deep_search', '{"query":{"table":"students","where":[{"field":"year","value":"1"}]}}'),
            ('direct_answer', '{"final_answer":["s05","s06"]}'),
        ]
        for name, arguments in cases:
            envelope = '{"action":' + json.dumps(name) + ',"arguments":' + arguments + '}'
            for separator in (' ', ': '):
                for payload, kind in ((arguments, 'action_prefix_arguments'), (envelope, 'redundant_action_prefix')):
                    raw = name + separator + payload
                    with self.subTest(raw=raw):
                        self.assertFalse(parse_action(raw).success)
                        result = normalize_protocol_output(raw)
                        self.assertTrue(result.normalization_applied)
                        self.assertEqual(result.raw_output, raw)
                        self.assertEqual(result.normalization_type, kind)
                        self.assertIn(payload, result.normalized_output)
                        parsed = parse_action(result.normalized_output)
                        self.assertTrue(parsed.success)
                        self.assertEqual(parsed.parsed_action, json.loads(envelope))
                        self.assertEqual(normalize_protocol_output(raw), result)

    def test_reject_ambiguous_or_non_protocol_text(self):
        envelope = action('lookup', entity='s00', field='advisor')
        cases = [
            'lookup ' + action('search', query='Student'),
            'unknown_action {"foo":"bar"}', 'plain {"entity":"s00","field":"advisor"}',
            'Here is ' + envelope, envelope + ' done', 'lookup {} trailing',
            'lookup {} {}', 'lookup []', 'lookup {', 'lookup{"entity":"s00"}',
            'lookup {"action":"lookup"}',
            'lookup {"entity":"s00","entity":"s01","field":"advisor"}',
            'deep_search {"query":{"table":"students","table":"professors"}}',
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                result = normalize_protocol_output(raw)
                self.assertFalse(result.normalization_applied)
                self.assertIsNone(result.normalized_output)
                self.assertFalse(parse_action(raw).success)

    def test_schema_still_rejects_without_structural_or_semantic_repair(self):
        for raw in ('lookup {"entity":"s00"}',
                    'lookup {"entity":"s00","field":"invented"}',
                    'deep_search {"query":{"table":"students","where":{"field":"year","value":"1"}}}',
                    'direct_answer {"final_answer":NaN}'):
            with self.subTest(raw=raw):
                result = normalize_protocol_output(raw)
                self.assertTrue(result.normalization_applied)
                self.assertFalse(parse_action(result.normalized_output).success)


class NormalizedRolloutTests(unittest.TestCase):
    def run_outputs(self, outputs):
        policy = ScriptPolicy(outputs)
        trajectory = rollout({'task_id': 'test', 'question': 'What is the advisor ID of student s00?'},
                             policy, ToolEnvironment(generate_kb()), RolloutConfig(max_steps=4, max_tool_calls=3))
        self.assertEqual(trajectory.raw_outputs, outputs)
        return trajectory, policy

    def test_strict_success_never_calls_normalizer(self):
        for raw in (action('direct_answer', final_answer='p03'),
                    '```json\n' + action('direct_answer', final_answer='p03') + '\n```'):
            with patch('src.rollouts.normalize_protocol_output') as normalizer:
                trajectory, _ = self.run_outputs([raw])
                normalizer.assert_not_called()
            status = trajectory.parse_status[0]
            self.assertTrue(status['strict_success'])
            self.assertFalse(status['normalization_applied'])
            self.assertEqual(status['parse_outcome'], 'strict_success')

    def test_normalized_tool_and_answer_without_model_repair(self):
        outputs = ['lookup {"entity":"s00","field":"advisor"}',
                   'direct_answer: {"final_answer":"p03"}']
        trajectory, policy = self.run_outputs(outputs)
        self.assertEqual(trajectory.termination_reason, 'direct_answer')
        self.assertEqual(trajectory.tool_sequence, ['lookup', 'direct_answer'])
        for status, raw in zip(trajectory.parse_status, outputs):
            self.assertEqual(status['raw_output'], raw)
            self.assertTrue(status['normalized_success'])
            self.assertFalse(status['strict_success'])
            self.assertFalse(status['repair_success'])
            self.assertEqual(status['retry_count'], 0)
            self.assertEqual(status['parse_outcome'], 'normalized_success')
        self.assertNotIn('format_feedback', policy.sessions[0].states[1].history[0])

    def test_repaired_output_can_be_strict_or_normalized(self):
        for repaired in (action('direct_answer', final_answer='p03'),
                         'direct_answer: {"final_answer":"p03"}'):
            trajectory, policy = self.run_outputs(['lookup {"entity":"s00"}', repaired])
            first, second = trajectory.parse_status
            self.assertTrue(first['normalization_applied'])
            self.assertTrue(first['invalid_arguments'])
            self.assertFalse(first['normalized_success'])
            self.assertEqual(first['parse_outcome'], 'repair_pending')
            self.assertEqual(second['parse_outcome'], 'repair_success')
            self.assertTrue(second['repair_success'])
            self.assertEqual(second['retry_count'], 1)
            self.assertEqual(second['normalized_success'], repaired.startswith('direct_answer:'))
            self.assertIn('format_feedback', policy.sessions[0].states[1].history[0])

    def test_final_invalid_does_not_execute_tools(self):
        raw = 'lookup ' + action('search', query='Student')
        trajectory, _ = self.run_outputs([raw, raw])
        self.assertEqual(trajectory.termination_reason, 'invalid_action')
        self.assertEqual(trajectory.tool_calls, [])
        self.assertEqual(trajectory.parse_status[-1]['parse_outcome'], 'final_invalid_action')
        self.assertFalse(trajectory.parse_status[-1]['normalization_applied'])


if __name__ == '__main__':
    unittest.main()
