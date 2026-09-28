import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.environment import ToolEnvironment
from src.kb import generate_kb
from src.tasks import generate_tasks, public_task
from src.policies import PolicyOutput
from src.policies.parser import parse_action
from src.policies.schema import build_messages, ACTION_SCHEMA
from src.rollouts import rollout, rollout_group, RolloutConfig
from src.evaluation.policy import evaluate_policy


def action(name, **arguments): return json.dumps({'action':name, 'arguments':arguments})


class ScriptPolicy:
    model_name = 'test-only-script'
    sampling_config = {'temperature':0.8}
    def __init__(self, outputs):
        self.outputs = outputs
        self.sessions = []
    def new_session(self, seed):
        session = copy.copy(self)
        session.states, session.seed, session.index = [], seed, 0
        self.sessions.append(session)
        return session
    def act(self, state):
        self.states.append(state)
        output = self.outputs[min(self.index, len(self.outputs)-1)]
        self.index += 1
        return PolicyOutput(output)


class ParserTests(unittest.TestCase):
    def test_public_schema_matches_kb_structure(self):
        from src.kb import RELATIONS
        from src.policies.schema import FIELDS, RELATIONS as PUBLIC_RELATIONS
        kb = generate_kb()
        for table, rows in kb.items():
            self.assertTrue(all(set(row) == set(FIELDS[table]) for row in rows.values()))
        self.assertEqual(RELATIONS, {(table, field): target for table, fields in PUBLIC_RELATIONS.items() for field, target in fields.items()})

    def test_valid_actions(self):
        cases = [action('lookup', entity='s00', field='advisor'), action('search', query='Student'),
                 action('deep_search', query={'table':'students','select':'advisor.department'}),
                 action('direct_answer', final_answer=['s00', 's01'])]
        for raw in cases:
            with self.subTest(raw=raw): self.assertTrue(parse_action(raw).success)
        self.assertTrue(parse_action('```json\n'+cases[0]+'\n```').success)

    def test_malformed(self):
        for raw in ['{', '{} {}', 'Here is JSON: {}', '{"action":"direct_answer", "action":"lookup", "arguments":{}}',
                    action('direct_answer', final_answer=float('nan'))]:
            self.assertFalse(parse_action(raw).success)

    def test_invalid_tool(self):
        result = parse_action(action('shell', command='ls'))
        self.assertFalse(result.success)
        self.assertEqual(result.error, 'Unknown action')

    def test_invalid_arguments(self):
        cases = [action('lookup', entity='s00'), action('lookup', entity=[], field='advisor'),
                 action('search', query={'table':'students','select':'advisor.department'}),
                 action('deep_search', query='who is the advisor'),
                 action('deep_search', query={'table':'students','limit':100}),
                 action('deep_search', query={'table':'students','where':[{'field':'year','op':'drop','value':2}]}),
                 action('deep_search', query={'table':['students']}),
                 action('direct_answer', final_answer={'answer':'p00'})]
        for raw in cases:
            result = parse_action(raw)
            self.assertFalse(result.success, raw)
            self.assertTrue(result.invalid_arguments, raw)


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.kb = generate_kb()
        self.tasks = generate_tasks(self.kb)
        self.task = public_task(self.tasks[0])
        self.factory = lambda: ToolEnvironment(self.kb)

    def test_normal_termination_and_no_leak(self):
        policy = ScriptPolicy([action('lookup', entity='s00', field='advisor'), action('direct_answer', final_answer='p03')])
        t = rollout(self.task, policy, self.factory(), RolloutConfig())
        self.assertEqual(t.termination_reason, 'direct_answer')
        self.assertFalse(t.truncated)
        self.assertEqual(t.number_of_steps, 2)
        self.assertEqual(t.total_cost, 1)
        self.assertEqual(t.tool_sequence, ['lookup', 'direct_answer'])
        self.assertEqual(t.token_ids, [])
        state = policy.sessions[0].states[-1]
        self.assertEqual(set(vars(state)), {'question', 'history', 'action_schema'})
        prompt = json.dumps(build_messages(state)).lower()
        for forbidden in ('ground_truth', 'query_plan', 'underlying_facts', 'total_cost', 'lambda', '"cost"'):
            self.assertNotIn(forbidden, prompt)
        self.assertEqual(state.history[0]['observation']['value'], 'p03')
        with self.assertRaises(ValueError): rollout(self.tasks[0], policy, self.factory(), RolloutConfig())

    def test_max_steps(self):
        t = rollout(self.task, ScriptPolicy([action('lookup', entity='s00', field='advisor')]), self.factory(), RolloutConfig(max_steps=2))
        self.assertTrue(t.truncated)
        self.assertEqual(t.termination_reason, 'max_steps')
        self.assertEqual(t.number_of_steps, 2)
        self.assertIsNone(t.final_answer)
        self.assertEqual(len(t.tool_calls), 2)

    def test_tool_limit_allows_final_but_blocks_next_tool(self):
        policy = ScriptPolicy([action('lookup', entity='s00', field='advisor')])
        t = rollout(self.task, policy, self.factory(), RolloutConfig(max_tool_calls=1))
        self.assertEqual(t.termination_reason, 'max_tool_calls')
        self.assertEqual(t.total_cost, 1)
        policy = ScriptPolicy([action('lookup', entity='s00', field='advisor'), action('direct_answer', final_answer='p03')])
        t = rollout(self.task, policy, self.factory(), RolloutConfig(max_tool_calls=1))
        self.assertEqual(t.termination_reason, 'direct_answer')

    def test_format_repair(self):
        policy = ScriptPolicy(['{', action('direct_answer', final_answer='unknown')])
        t = rollout(self.task, policy, self.factory(), RolloutConfig())
        self.assertEqual(t.termination_reason, 'direct_answer')
        self.assertEqual([p['retry_count'] for p in t.parse_status], [0, 1])
        self.assertEqual(t.raw_outputs[0], '{')
        self.assertEqual(t.number_of_steps, 1)
        self.assertIn('format_feedback', policy.sessions[0].states[1].history[0])

    def test_invalid_recorded(self):
        t = rollout(self.task, ScriptPolicy(['not JSON']), self.factory(), RolloutConfig())
        self.assertEqual(t.termination_reason, 'invalid_action')
        self.assertEqual(t.steps[-1]['action'], 'invalid_action')
        self.assertEqual(len(t.raw_outputs), 2)
        self.assertEqual(t.total_cost, 0)
        self.assertIsNone(t.final_answer)
        summary = evaluate_policy(self.tasks, [t])
        self.assertEqual(summary['aggregate']['invalid_action_rate'], 1)
        self.assertEqual(summary['aggregate']['normalized_accuracy'], 0)
        self.assertEqual(summary['diversity'][t.task_id]['unique_final_answers'], 0)

    def test_failed_proposals_do_not_disappear_from_preference_counts(self):
        policy = ScriptPolicy([action('deep_search', query='SELECT * FROM students')])
        t = rollout(self.task, policy, self.factory(), RolloutConfig())
        summary = evaluate_policy(self.tasks, [t])
        self.assertEqual(summary['proposed_action_counts']['deep_search'], 2)
        self.assertEqual(summary['first_proposed_action_counts']['deep_search'], 1)
        self.assertEqual(summary['tool_call_usage']['deep_search']['count'], 0)
        self.assertEqual(summary['aggregate']['invalid_action_rate'], 1)

    def test_group_isolation_and_repeatability(self):
        policy = ScriptPolicy([action('lookup', entity='s00', field='advisor'), action('direct_answer', final_answer='p03')])
        envs = []
        def factory():
            env = self.factory(); envs.append(env); return env
        group = rollout_group(self.task, policy, factory, G=4)
        self.assertEqual(len({s.seed for s in policy.sessions}), 4)
        self.assertTrue(all(s.states[0].history == [] for s in policy.sessions))
        self.assertEqual(len({id(e.engine.kb) for e in envs}), 4)
        self.assertTrue(all(e.engine.kb == self.kb for e in envs))
        group[0].observations[0]['value'] = 'modified'
        self.assertEqual(group[1].observations[0]['value'], 'p03')
        self.assertEqual(self.kb['students']['s00']['advisor'], 'p03')
        again = rollout_group(self.task, policy, factory, G=4)
        self.assertEqual([t.sampling_config['seed'] for t in group], [t.sampling_config['seed'] for t in again])

    def test_tool_execution_error_is_not_parse_error(self):
        policy = ScriptPolicy([action('lookup', entity='s99', field='advisor'), action('direct_answer', final_answer=None)])
        t = rollout(self.task, policy, self.factory(), RolloutConfig())
        summary = evaluate_policy(self.tasks, [t])
        self.assertEqual(summary['aggregate']['tool_execution_errors'], 1)
        self.assertEqual(summary['aggregate']['invalid_action_rate'], 0)
        self.assertEqual(t.total_cost, 1)

    def test_statistics_include_truncations_and_diversity(self):
        direct = rollout(self.task, ScriptPolicy([action('direct_answer', final_answer='p03')]), self.factory(), RolloutConfig(), 0)
        tool = rollout(self.task, ScriptPolicy([action('lookup', entity='s00', field='advisor')]), self.factory(), RolloutConfig(max_steps=1), 1)
        summary = evaluate_policy(self.tasks, [direct, tool])
        self.assertEqual(summary['aggregate']['normalized_accuracy'], 0.5)
        self.assertEqual(summary['aggregate']['truncation_rate'], 0.5)
        self.assertEqual(summary['aggregate']['direct_answer_without_tool_rate'], 0.5)
        self.assertEqual(summary['diversity'][self.task['task_id']]['unique_tool_sequences'], 2)
        self.assertEqual(summary['diversity'][self.task['task_id']]['cost_variance'], 0.25)


if __name__ == '__main__': unittest.main()
