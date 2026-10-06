import json
from copy import deepcopy
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env_v1 import Environment, generate_kb, eligibility, matching_students
from src.env_v1.tasks import generate_tasks, initial_state
from src.env_v1.evaluation import evaluate, summarize
from src.env_v1.protocol import parse_action, ACTION_SCHEMA
from src.tasks import public_task
from src.rewards import CostReward, CostAwareReward, MockAIReward
from src.rollouts import rollout, RolloutConfig
from src.policies import PolicyOutput
from src.policies.schema import build_messages
from src.policies import AgentState


def reference(task, env):
    """Feasibility fixture ONLY, never reported as a model baseline."""
    env.reset(public_task(task))
    spec = task['evaluation']
    if task['difficulty'] == 'L1':
        if spec['kind'] == 'lookup': answer = env.step('lookup', entity=spec['entity'])['record'][spec['field']]
        elif spec['kind'] == 'count': answer = env.step('aggregate', query=spec['query'], operation='count')['answer']
        else: answer = env.step('search', query=spec['query'])['answer']
    elif spec['kind'] == 'state' or 'scholarship_id' in spec.get('query', {}):
        sc = spec.get('scholarship_id', spec.get('query', {}).get('scholarship_id'))
        scholarship = env.step('lookup', entity=sc)['record']
        candidates = env.step('search', query=dict(table='enrollments', where=[dict(field='course_id', value=scholarship['required_course'])], select='student_id'))['answer']
        checks = env.step('verify', student_ids=candidates, scholarship_id=sc)['results']
        answer = [c['student_id'] for c in checks if c['eligible']]
        if spec['kind'] == 'state':
            pending = [c['student_id'] for c in checks if c['eligible'] and not c['already_applied']]
            env.step('submit', student_ids=pending, scholarship_id=sc)
            env.step('verify', student_ids=answer, scholarship_id=sc)
    else:
        q = spec['query']
        students = env.step('search', query=dict(table='students', where=[dict(field='year', value=q['year'])]))['answer']
        completed = env.step('search', query=dict(table='enrollments', where=[dict(field='course_id', value=q['course_id']), dict(field='status', value='completed')], select='student_id'))['answer']
        high_gpa = env.step('search', query=dict(table='students', where=[dict(field='gpa', op='gte', value=q['min_gpa'])]))['answer']
        answer = sorted(set(students) & set(completed) & set(high_gpa))
    env.step('direct_answer', final_answer=answer)
    return evaluate(task, env)


class TestV1(unittest.TestCase):
    def setUp(self):
        self.kb = generate_kb()
        self.tasks = generate_tasks(self.kb)
        self.env = Environment(self.kb)
        self.env.reset(public_task(self.tasks[0]))

    def test_data_and_tasks(self):
        self.assertEqual(self.kb, generate_kb())
        self.assertNotEqual(self.kb, generate_kb(43))
        self.assertEqual([len(self.kb[t]) for t in ('students','courses','enrollments','scholarships')], [30,12,90,5])
        self.assertEqual([sum(t['difficulty'] == l for t in self.tasks) for l in ('L1','L2','L3')], [10,15,15])
        for t in self.tasks:
            self.assertEqual(set(public_task(t)), {'task_id','question'})
            self.assertNotIn('expected_answer', t)
            if t['difficulty'] == 'L3':
                state = initial_state(self.kb, t)
                checks = [eligibility(state, sid, t['evaluation']['scholarship_id']) for sid in state['students']]
                self.assertGreater(sum(c['eligible'] for c in checks), 1)
                self.assertTrue(any(not c['eligible'] for c in checks))
                self.assertTrue(any(c['eligible'] and c['already_applied'] for c in checks))
                self.assertTrue(any(c['eligible'] and not c['already_applied'] for c in checks))

    def test_search_lookup_aggregate(self):
        self.assertEqual(self.env.step('lookup', entity='s00')['record'], self.kb['students']['s00'])
        q = dict(table='students', where=[dict(field='year', value=1)])
        expected = [s for s,r in self.kb['students'].items() if r['year'] == 1]
        self.assertEqual(self.env.step('search', query=q)['answer'], expected)
        self.assertEqual(self.env.step('aggregate', query=q, operation='count')['answer'], len(expected))
        q = dict(table='students', select='gpa')
        values = [s['gpa'] for s in self.kb['students'].values()]
        self.assertAlmostEqual(self.env.step('aggregate', query=q, operation='average')['answer'], sum(values)/len(values))
        self.assertEqual(self.env.step('aggregate', query=q, operation='max')['answer'], max(values))
        self.assertEqual(self.env.step('aggregate', query=q, operation='list')['answer'], list(dict.fromkeys(values)))

    def test_relational_and_verifier(self):
        self.assertEqual(self.env.step('deep_search', query=dict(scholarship_id='sc00'))['answer'], ['s00','s01','s02'])
        self.assertEqual(self.env.step('deep_search', query=dict(year=1,min_gpa=3.65,course_id='c00'))['answer'], ['s00','s01'])
        self.assertFalse(eligibility(self.kb, 's04','sc00')['eligible']) # other completed course cannot satisfy c00
        self.assertFalse(eligibility(self.kb, 's03','sc00')['eligible'])
        self.assertFalse(eligibility(self.kb, 's06','sc00')['eligible'])
        result = self.env.step('verify', student_ids=['s00','s01','s03','s04'], scholarship_id='sc00')['results']
        self.assertEqual([r['eligible'] for r in result], [True,True,False,False])

    def test_submit_rejections_reset_costs(self):
        before = deepcopy(self.kb)
        self.env.step('submit', student_ids=['s00','s03','missing','s01','s01'], scholarship_id='sc00')
        self.assertEqual([v['type'] for v in self.env.violations], ['invalid_entity','duplicate_submit','ineligible_submit','duplicate_submit'])
        self.assertEqual(len(self.env.engine.kb['applications']), 6)
        self.assertEqual(self.kb, before)
        self.env.step('deep_search', query=dict(scholarship_id='sc00'))
        self.assertEqual(CostReward().score(self.env.trajectory.tool_calls), 7)
        self.assertEqual(CostAwareReward(MockAIReward()).breakdown(self.env.trajectory)['total_cost'], 7)
        self.env.reset(public_task(self.tasks[0]))
        self.assertEqual(self.env.engine.kb, before)
        self.assertEqual(self.env.violations, [])

    def test_final_assertions_and_incomplete(self):
        task = next(t for t in self.tasks if t['difficulty'] == 'L3')
        env = Environment(initial_state(self.kb,task))
        env.reset(public_task(task))
        env.step('direct_answer', final_answer='done')
        self.assertFalse(evaluate(task,env)['success'])
        self.assertTrue(reference(task, env)['success'])
        env.engine.kb['students']['s00']['gpa'] = 0
        self.assertFalse(evaluate(task,env)['success'])
        self.assertIn('illegal_state_mutation', [v['type'] for v in evaluate(task,env)['rule_violations']])
        reference(task, env)
        env.engine.kb['applications']['s00:sc00']['status'] = 'changed'
        self.assertFalse(evaluate(task,env)['success'])
        reference(task, env)
        env.trajectory.steps.pop()
        self.assertFalse(evaluate(task,env)['success'])

    def test_all_reference_paths(self):
        rows=[]
        for t in self.tasks:
            row=reference(t, Environment(initial_state(self.kb,t)))
            self.assertTrue(row['success'], t['task_id'])
            self.assertEqual(row['rule_violation_count'], 0)
            self.assertTrue((1 <= row['tool_calls'] <= 2) if t['difficulty']=='L1' else (3 <= row['tool_calls'] <= 5))
            rows.append(row)
        self.assertEqual(summarize(rows)['overall_success'], 1)
        self.assertIsNone(summarize(rows)['recovery_success'])

    def test_failures(self):
        for kind in ('timeout','partial_result'):
            env=Environment(self.kb, failure_rate=1, failure_type=kind)
            env.reset(public_task(self.tasks[0]))
            obs=env.step('lookup', entity='s00')
            self.assertEqual(env.failures[0]['type'],kind)
            self.assertEqual(env.trajectory.total_cost,1)
            if kind=='timeout': self.assertEqual(obs['error_type'],'timeout')
            else: self.assertNotIn('gpa',obs['record'])
            env.step('submit', student_ids=['s01'], scholarship_id='sc00')
            self.assertEqual('s01:sc00' in env.engine.kb['applications'],kind=='partial_result')
        env=Environment(self.kb, failure_rate=1, failure_type='partial_result')
        env.reset(public_task(self.tasks[0]))
        obs=env.step('search',query=dict(table='students'))
        self.assertNotIn('answer',obs)
        self.assertEqual(len(obs['results']),29)
        for rate in (0,.2,.5):
            traces=[]
            for _ in range(2):
                env=Environment(self.kb,failure_rate=rate,max_tool_calls=30)
                env.reset(public_task(self.tasks[0]))
                traces.append([env.step('lookup',entity='s00') for i in range(30)])
            self.assertEqual(*traces)
            if rate==0: self.assertEqual(env.failures,[])

    def test_recovery_and_invalid_calls(self):
        env=Environment(self.kb,failure_rate=1,failure_type='timeout')
        env.reset(public_task(self.tasks[0]))
        env.step('lookup',entity='s00')
        env.failure_rate=0
        env.step('direct_answer',final_answer=env.step('lookup',entity='s00')['record']['gpa'])
        self.assertTrue(evaluate(self.tasks[0],env)['recovery_success'])
        self.env.step('submit',student_ids=['s01'],scholarship_id='sc00',status='approved')
        self.assertNotIn('s01:sc00', self.env.engine.kb['applications'])
        self.assertEqual(self.env.violations[0]['type'],'illegal_state_mutation')
        self.assertIn('error',self.env.step('lookup',entity='missing'))
        self.assertIn('error',self.env.step('deep_search',query=dict(scholarship_id='missing')))
        for raw in ('{"action":"search","arguments":{"query":{"table":"students","select":"advisor.name"}}}', '{"action":"submit","arguments":{"student_ids":"s01","scholarship_id":"sc00"}}', '{"action":"lookup","action":"lookup","arguments":{"entity":"s00"}}'):
            self.assertFalse(parse_action(raw).success)
        with self.assertRaises(ValueError): self.env.reset(self.tasks[0])

    def test_rollout_protocol(self):
        class Policy:
            model_name='scripted-test-only'
            sampling_config={}
            def new_session(self,seed): self.i=0; return self
            def act(self,state):
                self.i+=1
                assert state.action_schema['version']=='env_v1'
                action=dict(action='lookup',arguments=dict(entity='s00')) if self.i==1 else dict(action='direct_answer',arguments=dict(final_answer=3.8))
                return PolicyOutput(json.dumps(action))
        rollout(public_task(self.tasks[0]),Policy(),self.env,RolloutConfig())
        self.assertTrue(evaluate(self.tasks[0],self.env)['success'])
        self.assertIn('submit', build_messages(AgentState('question',[],ACTION_SCHEMA))[0]['content'])
        self.assertEqual(self.env.trajectory.tool_sequence,['lookup','direct_answer'])

    def test_partial_each_tool_and_recovery_state(self):
        task = next(t for t in self.tasks if t['difficulty']=='L3')
        env=Environment(initial_state(self.kb,task),failure_rate=1,failure_type='partial_result')
        env.reset(public_task(task))
        self.assertEqual(env.step('deep_search',query=dict(scholarship_id='sc00'))['answer'],['s00','s01'])
        obs=env.step('verify',student_ids=['s00','s01'],scholarship_id='sc00')
        self.assertEqual(len(obs['results']),1)
        self.assertNotIn('answer',env.step('aggregate',query=dict(table='students'),operation='count'))
        env.step('submit',student_ids=['s01','s02'],scholarship_id='sc00')
        env.failure_rate=0
        status=env.step('verify',student_ids=['s00','s01','s02'],scholarship_id='sc00')
        self.assertTrue(all(r['already_applied'] for r in status['results']))
        env.step('direct_answer',final_answer='done')
        row=evaluate(task,env)
        self.assertTrue(row['recovery_success'])
        self.assertEqual(row['rule_violation_count'],0)

    def test_task_success_separate_from_attempt_violations(self):
        task=next(t for t in self.tasks if t['difficulty']=='L3')
        env=Environment(initial_state(self.kb,task))
        env.reset(public_task(task))
        env.step('submit',student_ids=['s00','s03','s01','s02'],scholarship_id='sc00')
        env.step('direct_answer',final_answer='done')
        row=evaluate(task,env)
        self.assertTrue(row['task_success'])
        self.assertEqual(row['rule_violation_count'],2)
        env.engine.kb['applications']['fake']=deepcopy(env.engine.kb['applications']['s00:sc00'])
        self.assertFalse(evaluate(task,env)['success'])

    def test_observation_copy_budget_and_empty_aggregate(self):
        obs=self.env.step('lookup',entity='s00')
        obs['record']['gpa']=0
        self.assertEqual(self.env.engine.kb['students']['s00']['gpa'],3.8)
        self.assertEqual(self.env.trajectory.observations[0]['record']['gpa'],3.8)
        self.assertIsNone(self.env.step('aggregate',query=dict(table='students',where=[dict(field='year',value=99)],select='gpa'),operation='average')['answer'])
        self.env.max_tool_calls=2
        with self.assertRaises(RuntimeError): self.env.step('lookup',entity='s00')
        self.env.step('direct_answer',final_answer=3.8)
        with self.assertRaises(RuntimeError): self.env.step('lookup',entity='s00')
        self.assertEqual(self.env.trajectory.total_cost,2)

    def test_wrong_answer_and_read_only_mutation(self):
        self.env.step('direct_answer',final_answer=[3.8])
        self.assertFalse(evaluate(self.tasks[0],self.env)['success'])
        self.env.reset(public_task(self.tasks[0]))
        self.env.step('submit',student_ids=['s01'],scholarship_id='sc00')
        self.env.step('direct_answer',final_answer=3.8)
        self.assertFalse(evaluate(self.tasks[0],self.env)['success'])
        for args in [dict(failure_rate=-1),dict(failure_rate=1.1),dict(failure_type='other')]:
            with self.assertRaises(ValueError): Environment(self.kb,**args)
