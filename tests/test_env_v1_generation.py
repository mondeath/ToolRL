"""Data-only expansion invariants; no GPU or training dependencies."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_model_scale import load_contract
from src.env_v1 import FIELDS, generate_kb, eligibility
from src.env_v1.tasks import generate_tasks, initial_state
from src.env_v1.generation import SPLIT_SEEDS, generate_instance
from src.evaluation.generator_audit import audit_splits, instance_fingerprint, non_id_signature


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = [(split, seed, *generate_instance(seed, split))
                        for split, seeds in SPLIT_SEEDS.items() for seed in seeds]
        cls.audits = audit_splits(cls.datasets)

    def test_frozen_contract_and_seed42_exact(self):
        load_contract()  # All existing file/KB/task/schema/prompt hashes.
        kb, tasks = generate_instance(42, 'diagnostic')
        self.assertEqual(kb, generate_kb(42))
        self.assertEqual(tasks, generate_tasks(kb))

    def test_all_1640_reference_paths_and_difficulty(self):
        self.assertEqual(sum(len(a['tasks']) for a in self.audits), 1640)
        for audit in self.audits:
            for task in audit['tasks']:
                self.assertTrue(task['reference_success'])
                self.assertEqual(task['reference_tool_calls'], {'L1': 1, 'L2': 3, 'L3': 5}[task['difficulty']])

    def test_schema_references_and_eligibility_structure(self):
        task_fields = set(generate_tasks(generate_kb())[0])
        for split, seed, kb, tasks in self.datasets:
            self.assertEqual(set(kb), set(FIELDS))
            for table, rows in kb.items():
                for key, row in rows.items():
                    self.assertEqual(key, row['id'])
                    self.assertEqual(set(row), set(FIELDS[table]))
            for row in kb['enrollments'].values():
                self.assertIn(row['student_id'], kb['students'])
                self.assertIn(row['course_id'], kb['courses'])
                self.assertEqual(row['grade'] is None, row['status'] != 'completed')
            for sc, row in kb['scholarships'].items():
                self.assertIn(row['required_course'], kb['courses'])
                checks = [eligibility(kb, sid, sc) for sid in kb['students']]
                self.assertEqual(sum(c['eligible'] for c in checks), 3)
                self.assertTrue(any(c['eligible'] and c['already_applied'] for c in checks))
                self.assertTrue(any(c['eligible'] and not c['already_applied'] for c in checks))
            for app in kb['applications'].values():
                self.assertTrue(eligibility(kb, app['student_id'], app['scholarship_id'])['eligible'])
            for task in tasks:
                self.assertEqual(set(task), task_fields |
                                 ({'initial_applications'} if task['difficulty'] == 'L3' else set()))

    def test_reproducibility_isolation_and_non_id_novelty(self):
        for split, seed, kb, tasks in self.datasets:
            self.assertEqual((kb, tasks), generate_instance(seed, split))
        self.assertEqual(len({a['kb_sha256'] for a in self.audits}), 41)
        self.assertEqual(len({a['non_id_signature_sha256'] for a in self.audits}), 41)
        fingerprints = [t['instance_sha256'] for a in self.audits for t in a['tasks']]
        self.assertEqual(len(set(fingerprints)), 1640)
        kb, tasks = generate_instance(1000, 'train')
        changed = deepcopy(tasks[0])
        changed['task_id'] = 'cosmetic-renaming'
        self.assertEqual(instance_fingerprint(kb, changed), instance_fingerprint(kb, tasks[0]))
        renamed = deepcopy(kb)
        renamed['students'] = {f'x{i}': dict(row, id=f'x{i}', name='renamed')
                               for i, row in enumerate(kb['students'].values())}
        self.assertEqual(non_id_signature(kb), non_id_signature(renamed))

    def test_requested_fields_and_real_outcomes_vary(self):
        worlds = [kb for split, seed, kb, tasks in self.datasets if split != 'diagnostic']
        for table, field in [('students', 'gpa'), ('students', 'year'), ('students', 'department'),
                             ('enrollments', 'course_id'), ('enrollments', 'status'),
                             ('scholarships', 'min_gpa'), ('scholarships', 'eligible_year'),
                             ('scholarships', 'required_course')]:
            self.assertGreater(len({tuple((key, row[field]) for key, row in sorted(kb[table].items()))
                                    for kb in worlds}), 1)
        self.assertGreater(len({repr(kb['applications']) for kb in worlds}), 1)
        for level in ('L1', 'L2', 'L3'):
            vectors = {tuple(t['outcome_sha256'] for t in a['tasks'] if t['difficulty'] == level)
                       for a in self.audits}
            self.assertGreater(len(vectors), 1)

    def test_invalid_split_seeds_fail_closed(self):
        for seed, split in [(42, 'train'), (1000, 'dev'), (2000, 'train'), (43, 'diagnostic'),
                            (999, 'train'), (1000.0, 'train'), (True, 'train'), (1000, 'other')]:
            with self.assertRaises(ValueError):
                generate_instance(seed, split)
        item = self.datasets[0]
        with self.assertRaisesRegex(ValueError, 'seed'):
            audit_splits([item, item])
        split, seed, kb, tasks = item
        clone = deepcopy(tasks)
        for task in clone:
            task['task_id'] = 'renamed-' + task['task_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate world'):
            audit_splits([item, ('train', 1001, kb, clone)])

    def test_initial_state_and_episode_isolation(self):
        from src.env_v1 import Environment
        from src.tasks import public_task
        kb, tasks = generate_instance(1000, 'train')
        task = next(t for t in tasks if t['difficulty'] == 'L3')
        original = deepcopy((kb, task))
        state = initial_state(kb, task)
        env = Environment(state)
        env.reset(public_task(task))
        sc = task['evaluation']['scholarship_id']
        pending = [sid for sid in state['students'] if eligibility(state, sid, sc)['eligible']
                   and not eligibility(state, sid, sc)['already_applied']]
        env.step('submit', student_ids=pending, scholarship_id=sc)
        self.assertNotEqual(env.engine.kb, state)
        env.reset(public_task(task))
        self.assertEqual(env.engine.kb, state)
        self.assertEqual((kb, task), original)


if __name__ == '__main__':
    unittest.main()
