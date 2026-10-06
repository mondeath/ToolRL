"""Reporting must preserve actual failures and label scripted probes separately."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import report_env_v1
import check_env_v1_failures
from src.env_v1.evaluation import summarize


class ReportTests(unittest.TestCase):
    def test_report_retains_low_success_and_no_fabricated_examples(self):
        rows=[]
        for level in ('L1','L2','L3'):
            rows.append(dict(task_id=level,difficulty=level,success=False,tool_cost=1,
                trajectory_length=2,tool_calls=1,expensive_tool_calls=0,rule_violation_count=0,
                recovery_success=None,rule_violations=[],trajectory=dict(question='test',tool_calls=[],
                raw_outputs=[],final_answer=[3.8],termination_reason='direct_answer')))
        summary=summarize(rows)
        summary['run']=dict(model_config=dict(revision='test-revision'),elapsed_seconds=1)
        text=report_env_v1.render(rows,summary)
        self.assertIn('| overall success | 0.0% |',text)
        self.assertIn('实际仅有 0 条可展示的成功轨迹',text)
        self.assertIn('低于 20%',text)
        self.assertIn('脚本 lookup 重试探针',text)
        self.assertEqual(text.count('### 失败：'),3)

    def test_failure_probe_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'results').mkdir()
            with patch.object(check_env_v1_failures,'ROOT',root/'experiment'),contextlib.redirect_stdout(io.StringIO()):
                check_env_v1_failures.run()
            data=json.loads((root/'results/env_v1_failure_smoke.json').read_text())
            self.assertEqual(data['kind'],'scripted_fault_retry_probe_not_model_baseline')
            self.assertEqual([r['failure_rate'] for r in data['results']],[0,.2,.5])
            self.assertEqual(data['results'][0]['injected_episodes'],0)
            for r in data['results'][1:]:
                self.assertGreater(r['failure_counts']['timeout'],0)
                self.assertGreater(r['failure_counts']['partial_result'],0)
                self.assertEqual(r['recovery_success'],1)

    def test_real_runner_writes_all_outcomes_with_mock_inference(self):
        import run_env_v1
        from src.policies import PolicyOutput
        class StubPolicy:
            model_name='test-only'
            sampling_config={}
            runtime_metadata={'device':'test'}
            def __init__(self,config): pass
            def new_session(self,seed): return self
            def act(self,state): return PolicyOutput('{"action":"direct_answer","arguments":{"final_answer":null}}')
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'test.jsonl'
            with patch.object(run_env_v1,'HuggingFacePolicy',StubPolicy),patch.object(sys,'argv',['run_env_v1','--output',str(output)]),contextlib.redirect_stdout(io.StringIO()):
                run_env_v1.main()
            rows=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows),40)
            self.assertTrue(all(not r['success'] for r in rows))
            summary=json.loads(output.with_suffix('.summary.json').read_text())
            self.assertEqual(summary['overall_success'],0)
            self.assertEqual(summary['run']['failure_rate'],0)
            self.assertEqual(len(summary['run']['prompt_sha256']),64)
            self.assertEqual(len(json.loads(output.with_suffix('.tasks.json').read_text())),40)
            self.assertTrue(output.with_suffix('.kb.json').exists())
            prompt=json.loads(output.with_suffix('.prompt.json').read_text())
            self.assertIn('FULL action formats',prompt[0]['content'])
            self.assertIn('final_answer\":3.5',prompt[0]['content'])
        for option,value in [('--task-count','0'),('--failure-rate','2')]:
            with patch.object(sys,'argv',['run_env_v1',option,value]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                run_env_v1.main()

    def test_protocol_comparison_is_separate_and_real(self):
        rows=[dict(task_id=level,difficulty=level,success=False,trajectory=dict(raw_outputs=['bad'],tool_calls=[],final_answer=None,termination_reason='invalid_action')) for level in ('L1','L2','L3')]
        summary=dict(overall_success=0,success_by_difficulty=dict(L1=0,L2=0,L3=0))
        text=report_env_v1.protocol_comparison(rows,summary,[dict(success=True)])
        self.assertIn('独立保存，不混入最终指标',text)
        self.assertIn('首轮 L3',text)
        self.assertIn('不是 RL 改进',text)

    def test_reference_examples_never_claim_model_success(self):
        rows=[dict(task_id=level,difficulty=level,success=True,tool_calls=3,tool_cost=3,rule_violation_count=0,assertion_results=[],trajectory=dict(tool_calls=[],final_answer=[])) for level in ('L2','L3')]
        text=report_env_v1.reference_examples(rows)
        self.assertIn('非 Qwen baseline',text)
        self.assertIn('完全不计入模型成功率',text)
        self.assertEqual(text.count('### 参考成功'),2)
