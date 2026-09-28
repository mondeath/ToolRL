import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kb import generate_kb, validate_kb
from src.tasks import generate_tasks, public_task
from src.tools import QueryEngine
from src.environment import ToolEnvironment
from src.rewards import MockAIReward, PureReward, CostAwareReward
from src.evaluation import evaluate, normalized_equal
from src.trajectories import save_jsonl


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.kb = generate_kb()
        self.tasks = generate_tasks(self.kb)
        self.env = ToolEnvironment(self.kb, top_k=2)
        self.env.reset(public_task(self.tasks[0]))

    def test_determinism_and_integrity(self):
        self.assertEqual(self.kb, generate_kb())
        self.assertNotEqual(self.kb, generate_kb(43))
        self.assertEqual([len(self.kb[k]) for k in ('students', 'courses', 'professors', 'projects', 'clubs')], [30, 8, 5, 5, 4])
        bad = copy.deepcopy(self.kb)
        bad['students']['s00']['advisor'] = 'missing'
        with self.assertRaises(ValueError): validate_kb(bad)
        bad = copy.deepcopy(self.kb)
        cid = bad['students']['s00']['courses'][0]
        bad['courses'][cid]['students'].remove('s00')
        with self.assertRaises(ValueError): validate_kb(bad)

    def test_tasks_independent_oracle(self):
        self.assertEqual(self.tasks, generate_tasks(self.kb))
        self.assertEqual(len(self.tasks), 64)
        self.assertEqual(len({t['question'] for t in self.tasks}), 64)
        for task in self.tasks:
            plan = task['evaluation']['query_plan']
            if task['task_type'] in ('single-hop', 'two-hop'):
                student = self.kb['students'][plan['where'][0]['value']]
                expected = student['advisor']
                if task['task_type'] == 'two-hop': expected = self.kb['professors'][expected]['department']
            else:
                expected = []
                for sid, student in sorted(self.kb['students'].items()):
                    if all(c['value'] in student[c['field']] if isinstance(student[c['field']], list) else c['value'] == student[c['field']] for c in plan['where']):
                        expected.append(sid)
            self.assertEqual(task['ground_truth'], expected)
        with self.assertRaises(ValueError): self.env.reset(self.tasks[0])

    def test_queries_and_top_k(self):
        engine = QueryEngine(self.kb)
        result = engine.search({'table': 'students'}, 2)
        self.assertEqual(len(result['results']), 2)
        self.assertTrue(result['truncated'])
        self.assertNotIn('evidence', result)
        self.assertEqual(len(engine.search('Student-', 2)['results']), 2)
        query = {'table':'students', 'where':[{'field':'id','value':'s00'}], 'select':'courses.professor.department'}
        expected = list(dict.fromkeys(self.kb['professors'][self.kb['courses'][cid]['professor']]['department'] for cid in self.kb['students']['s00']['courses']))
        self.assertEqual(engine.execute(query)['answer'], expected)
        with self.assertRaises(ValueError): engine.search(query, 2)
        # Changing KB changes query answers without changing any stored task truth.
        self.kb['students']['s00']['advisor'] = 'p04'
        self.assertEqual(engine.execute(self.tasks[0]['evaluation']['query_plan'])['answer'], 'p04')

    def test_accounting_errors_and_terminal(self):
        observation = self.env.step('lookup', entity='missing', field='advisor')
        self.assertIn('error', observation)
        good = self.env.step('lookup', entity='s00', field='advisor')
        good['value'] = 'tampered'
        self.assertNotEqual(self.env.trajectory.observations[-1]['value'], 'tampered')
        self.env.step('direct_answer', final_answer='p00')
        self.assertEqual(self.env.trajectory.total_cost, 2)
        self.assertEqual(len(self.env.trajectory.tool_calls), 2)
        with self.assertRaises(RuntimeError): self.env.step('search', query='student')
        self.env.reset(public_task(self.tasks[0]))
        self.env.max_tool_calls = 1
        self.env.step('lookup', entity='s00', field='advisor')
        with self.assertRaises(RuntimeError): self.env.step('search', query='student')
        self.env.step('direct_answer', final_answer='unknown')

    def test_rewards_serialization_evaluation(self):
        answer = self.env.step('lookup', entity='s00', field='advisor')['value']
        self.env.step('direct_answer', final_answer=answer)
        t = self.env.trajectory
        reward = CostAwareReward(MockAIReward(), 0.1)
        self.assertEqual(PureReward(MockAIReward()).score(t), 1)
        self.assertAlmostEqual(reward.score(t), 0.9)
        self.assertEqual(CostAwareReward(MockAIReward(), 0).score(t), 1)
        summary = evaluate(self.tasks, [t], reward)
        self.assertEqual(summary['normalized_accuracy'], 1)
        self.assertEqual(summary['average_tool_cost'], 1)
        self.assertEqual(summary['deep_search_usage_rate'], 0)
        self.assertTrue(normalized_equal([' P01 ', 'p00'], ['P00', 'p01']))
        self.assertFalse(normalized_equal([], ['p00']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trajectory.jsonl'
            save_jsonl(path, [t])
            self.assertEqual(json.loads(path.read_text()), t.to_dict())
        t.total_cost = 0
        with self.assertRaises(ValueError): reward.score(t)

    def test_zero_tool_episode(self):
        self.env.step('direct_answer', final_answer='unknown')
        summary = evaluate(self.tasks, [self.env.trajectory], CostAwareReward(MockAIReward()))
        self.assertEqual(summary['average_tool_cost'], 0)
        self.assertEqual(summary['deep_search_call_fraction'], 0)


if __name__ == '__main__': unittest.main()
