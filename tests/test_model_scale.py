"""Diagnostic routing and offline metrics; never load/download a model in tests."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import run_env_v1
import run_model_scale
from src.evaluation.model_scale import annotate, diagnostics, select_tasks, normalize_dtype, resolve_dtype
from src.env_v1 import generate_kb
from src.env_v1.tasks import generate_tasks
from src.policies import PolicyOutput


class ModelScaleTests(unittest.TestCase):
    def test_frozen_balanced_subset_and_drift_guard(self):
        contract, tasks = run_model_scale.load_contract()
        self.assertEqual([sum(t['difficulty']==level for t in tasks) for level in ('L1','L2','L3')], [5,5,5])
        self.assertEqual(len({t['task_id'] for t in tasks}), 15)
        self.assertEqual(len({m['revision'] for m in contract['models'].values()}), 3)
        with patch.object(run_model_scale, 'SYSTEM_PROMPT', run_model_scale.SYSTEM_PROMPT+' changed'):
            with self.assertRaisesRegex(ValueError, 'prompt/schema'):
                run_model_scale.load_contract()

    def test_select_ids_order_and_validation(self):
        tasks=generate_tasks(generate_kb())
        ids=['v1-L3-12', 'v1-L1-02']
        self.assertEqual([t['task_id'] for t in select_tasks(tasks, ids)], ids)
        self.assertEqual(len(select_tasks(tasks, ids, 1)), 1)
        for bad in ([], ['missing'], ids+ids, 'not a list'):
            with self.assertRaises(ValueError): select_tasks(tasks, bad)
        with self.assertRaises(ValueError): select_tasks(tasks, count=0)

    def test_repetitions_are_adjacent_executed_calls_only(self):
        a=dict(tool_name='deep_search', arguments={'a':1,'b':2})
        b=dict(tool_name='lookup', arguments={'entity':'s00'})
        reordered=dict(tool_name='deep_search',arguments={'b':2,'a':1})
        row=dict(trajectory=dict(tool_calls=[a,reordered,a,b,a], termination_reason='max_tool_calls',
                                steps=[dict(action='blocked_tool_call', proposed_action=a)]))
        original=deepcopy(row)
        output=annotate(row)
        self.assertEqual(row, original)
        self.assertEqual(output['repeated_identical_action_count'],2)
        self.assertEqual(output['deep_search_calls'],4)
        self.assertIsNone(output['answer_ready_but_continued'])
        self.assertEqual(diagnostics([row])['max_tool_call_rate'],1)
        row['trajectory']['tool_calls']=[a,b,a]
        self.assertEqual(annotate(row)['repeated_identical_action_count'],0)
        with self.assertRaises(ValueError): diagnostics([])

    def test_model_override_keeps_generation_and_task_order(self):
        seen=[]
        class StubPolicy:
            sampling_config={}
            def __init__(self,config):
                seen.append(deepcopy(config))
                self.model_name=config['model_name']
                self.runtime_metadata={'device':config['device'],'dtype':config['dtype']}
            def new_session(self,seed): return self
            def act(self,state): return PolicyOutput('{"action":"direct_answer","arguments":{"final_answer":null}}')
        contract,tasks=run_model_scale.load_contract()
        with tempfile.TemporaryDirectory() as folder:
            for model in contract['models']:
                output=Path(folder)/contract['models'][model]['directory']/'results.jsonl'
                argv=['run_env_v1','--model',model,'--device','cuda','--task-ids-file',str(ROOT/'configs/env_v1_diagnostic_tasks.json'),'--output',str(output)]
                with patch.object(sys,'argv',argv),patch.object(run_env_v1,'HuggingFacePolicy',StubPolicy),contextlib.redirect_stdout(io.StringIO()):
                    run_env_v1.main()
                rows=[json.loads(l) for l in output.read_text().splitlines()]
                self.assertEqual([r['task_id'] for r in rows],[t['task_id'] for t in tasks])
                for r in rows:
                    self.assertIn('termination_reason',r)
                    self.assertIn('repeated_identical_action_count',r)
                self.assertEqual(seen[-1]['revision'],contract['models'][model]['revision'])
                self.assertFalse(seen[-1]['cpu_fallback'])
                self.assertEqual(seen[-1]['dtype'], 'bfloat16')
                for k,v in contract['frozen']['policy'].items(): self.assertEqual(seen[-1][k],v)
                summary=json.loads(output.with_name('summary.json').read_text())
                run_model_scale.validate_run(summary,contract,tasks,model,'cuda')
        for value in ('unknown', 'Qwen/Qwen2.5-14B-Instruct'):
            with patch.object(sys,'argv',['run_env_v1','--model',value]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                run_env_v1.main()

    def test_archive_reuse_and_pending_report(self):
        contract,tasks=run_model_scale.load_contract()
        model='Qwen/Qwen2.5-1.5B-Instruct'
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            directory=output/contract['models'][model]['directory']
            summary=run_model_scale.reuse_baseline(directory,contract,tasks)
            self.assertEqual(summary['task_count'],15)
            self.assertEqual(summary['overall_success'],1/15)
            self.assertEqual(summary['diagnostics']['repeated_identical_action_count'],78)
            self.assertNotIn('elapsed_seconds',summary['run'])
            self.assertTrue((directory/'run.log').exists())
            run_model_scale.validate_run(summary,contract,tasks,model,'cpu')
            with self.assertRaisesRegex(ValueError,'device differs'):
                run_model_scale.validate_run(summary,contract,tasks,model,'cuda')
            for key,value in [('do_sample',True),('max_new_tokens',512),('dtype','float16')]:
                bad=deepcopy(summary);bad['run']['model_config'][key]=value
                with self.assertRaisesRegex(ValueError,'Dtype' if key=='dtype' else 'Generation'):
                    run_model_scale.validate_run(bad,contract,tasks,model)
            with self.assertRaisesRegex(ValueError,'Task order'):
                run_model_scale.validate_run(summary,contract,list(reversed(tasks)),model)
            report=output/'report.md'
            run_model_scale.write_report(output,contract,tasks,report)
            text=report.read_text()
            self.assertIn('6.7%',text)
            self.assertEqual(text.count('待云端运行'),2)
            self.assertIn('证据尚不足',text)
            self.assertIn('null 不是 false',text)
            bad=deepcopy(summary);bad['overall_success']=1
            (directory/'summary.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'Summary does not match'):
                run_model_scale.write_report(output,contract,tasks,report)

    def test_cloud_commands_are_uniform_and_partial_outputs_protected(self):
        contract,tasks=run_model_scale.load_contract()
        commands=[]
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            template_dir=base/'template'
            template=run_model_scale.reuse_baseline(template_dir,contract,tasks)
            data=(template_dir/'results.jsonl').read_text()
            def fake_run(command, **kwargs):
                commands.append(command)
                self.assertTrue(kwargs['check'])
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
                model=command[command.index('--model')+1]
                destination=Path(command[command.index('--output')+1])
                summary=deepcopy(template)
                summary.pop('provenance')
                summary['run']['model_config'].update(model_name=model,revision=contract['models'][model]['revision'],device='cuda')
                dtype=command[command.index('--dtype')+1]
                self.assertEqual(dtype,'bfloat16')
                summary['run']['model_config']['dtype']=dtype
                summary['run']['runtime'].update(device='cuda',dtype=dtype)
                destination.write_text(data)
                destination.with_name('summary.json').write_text(json.dumps(summary))
            argv=['run_model_scale','--device','cuda','--dtype','bf16','--output-root',str(base/'cloud')]
            with patch.object(sys,'argv',argv),patch.object(run_model_scale.subprocess,'run',side_effect=fake_run),patch.object(run_model_scale,'write_report') as report:
                run_model_scale.main()
                report.assert_called_once()
            self.assertEqual(len(commands),3)
            for command in commands:
                self.assertEqual(command[command.index('--device')+1],'cuda')
                self.assertEqual(command[command.index('--failure-rate')+1],'0')
                self.assertEqual(command[command.index('--task-ids-file')+1],str(ROOT/'configs/env_v1_diagnostic_tasks.json'))
            with patch.object(sys,'argv',argv),patch.object(run_model_scale.subprocess,'run') as process,patch.object(run_model_scale,'write_report'):
                run_model_scale.main()
                process.assert_not_called()
            partial=base/'partial'/contract['models']['Qwen/Qwen2.5-1.5B-Instruct']['directory']
            partial.mkdir(parents=True)
            (partial/'results.jsonl').write_text('keep partial output\n')
            with patch.object(sys,'argv',['run_model_scale','--output-root',str(base/'partial')]),self.assertRaisesRegex(ValueError,'Partial output'):
                run_model_scale.main()
            self.assertEqual((partial/'results.jsonl').read_text(),'keep partial output\n')

    def test_download_uses_pinned_model_revisions_without_inference(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        download=Mock()
        with patch.dict(sys.modules, {'huggingface_hub':SimpleNamespace(snapshot_download=download)}),patch.object(sys,'argv',['run_model_scale','--download-only']):
            run_model_scale.main()
        contract,_=run_model_scale.load_contract()
        self.assertEqual(download.call_count,3)
        for call,(model,spec) in zip(download.call_args_list,contract['models'].items()):
            self.assertEqual(call.args[0],model)
            self.assertEqual(call.kwargs['revision'],spec['revision'])


    def test_dtype_defaults_aliases_and_cpu_guard(self):
        self.assertEqual(resolve_dtype('cuda'), 'bfloat16')
        self.assertEqual(resolve_dtype('cpu'), 'float32')
        self.assertEqual(resolve_dtype('cuda', 'bf16'), 'bfloat16')
        self.assertEqual(resolve_dtype('cuda', 'fp32'), 'float32')
        for dtype in ('float16', 'int8'):
            with self.assertRaises(ValueError): resolve_dtype('cuda', dtype)
        for module in (run_env_v1, run_model_scale):
            with patch.object(sys,'argv',['runner','--device','cpu','--dtype','bf16']),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                module.main()

    def test_float32_archive_cannot_be_reused_as_bfloat16(self):
        contract,tasks=run_model_scale.load_contract()
        model='Qwen/Qwen2.5-1.5B-Instruct'
        with tempfile.TemporaryDirectory() as folder:
            summary=run_model_scale.reuse_baseline(Path(folder),contract,tasks)
            with self.assertRaisesRegex(ValueError,'Runtime dtype differs'):
                run_model_scale.validate_run(summary,contract,tasks,model,dtype='bfloat16')
            summary['run']['model_config']['dtype']='bfloat16'
            with self.assertRaisesRegex(ValueError,'configuration/runtime mismatch'):
                run_model_scale.validate_run(summary,contract,tasks,model)
