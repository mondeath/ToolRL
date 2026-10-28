"""Fixed model-size diagnostic: archive reuse/preparation, cloud runs, offline report."""
from _common import ROOT
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from src.env_v1 import generate_kb
from src.env_v1.tasks import generate_tasks
from src.env_v1.protocol import ACTION_SCHEMA, SYSTEM_PROMPT
from src.env_v1.evaluation import summarize
from src.evaluation.model_scale import annotate, diagnostics, digest, select_tasks, normalize_dtype, resolve_dtype


def load_contract():
    contract = json.loads((ROOT/'configs/env_v1_model_scale.json').read_text())
    ids = json.loads((ROOT/'configs/env_v1_diagnostic_tasks.json').read_text())['task_ids']
    kb, all_tasks = generate_kb(), generate_tasks(generate_kb())
    frozen = contract['frozen']
    if digest(kb) != frozen['kb_sha256'] or digest(all_tasks) != frozen['all_tasks_sha256']:
        raise ValueError('Frozen KB/task set changed')
    if hashlib.sha256((SYSTEM_PROMPT+json.dumps(ACTION_SCHEMA, sort_keys=True)).encode()).hexdigest() != frozen['prompt_sha256']:
        raise ValueError('Frozen prompt/schema changed')
    for relative, expected in frozen['source_sha256'].items():
        actual = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        reviewed_normalizer_fix = (relative == 'src/policies/normalizer.py' and
            actual == 'f3cf012677be677cab32f9608edc44c69ea5ae01141af5dbd2e349706d81174a')
        if actual != expected and not reviewed_normalizer_fix:
            raise ValueError('Frozen behavior source changed: ' + relative)
    tasks = select_tasks(all_tasks, ids)
    if len(tasks) != 15 or any(sum(t['difficulty'] == level for t in tasks) != 5 for level in ('L1','L2','L3')):
        raise ValueError('Diagnostic subset must contain exactly 5 tasks per level')
    return contract, tasks


def validate_run(summary, contract, tasks, model, device=None, dtype=None):
    run, frozen = summary['run'], contract['frozen']
    if run['failure_rate'] != 0 or run['rollout_config'] != frozen['rollout']:
        raise ValueError('Failure/rollout settings changed')
    config = run['model_config']
    if any(config.get(k) != v for k, v in frozen['policy'].items()):
        raise ValueError('Generation settings changed')
    actual_dtype = run['runtime']['dtype']
    if config.get('dtype') not in ('float32', 'bfloat16') or actual_dtype != config['dtype']:
        raise ValueError('Dtype configuration/runtime mismatch')
    if dtype and actual_dtype != normalize_dtype(dtype):
        raise ValueError('Runtime dtype differs; use a new output directory for a uniform precision run')
    if config['model_name'] != model or config['revision'] != contract['models'][model]['revision']:
        raise ValueError('Wrong model/revision')
    if device and run['runtime']['device'] != device:
        raise ValueError('Runtime device differs; use a new output directory for a uniform cloud run')
    if run['kb_sha256'] != frozen['kb_sha256'] or run['prompt_sha256'] != frozen['prompt_sha256']:
        raise ValueError('KB/prompt changed')
    if run['tasks_sha256'] != digest(tasks):
        raise ValueError('Task order or contents differ')


def reuse_baseline(destination, contract, tasks):
    """Filter archived episodes without re-evaluating or pretending they ran on CUDA."""
    model = 'Qwen/Qwen2.5-1.5B-Instruct'
    source = ROOT/'results/env_v1_baseline.jsonl'
    summary = json.loads(source.with_suffix('.summary.json').read_text())
    all_tasks = json.loads(source.with_suffix('.tasks.json').read_text())
    validate_run(summary, contract, all_tasks, model, device='cpu', dtype='float32')
    if digest(all_tasks) != contract['frozen']['all_tasks_sha256']:
        raise ValueError('Archived full task set changed')
    indexed = {r['task_id']: r for r in map(json.loads, source.read_text().splitlines())}
    if len(indexed) != 40:
        raise ValueError('Incomplete archived baseline')
    for task in tasks:
        if task != next(t for t in all_tasks if t['task_id'] == task['task_id']):
            raise ValueError('Archived task contents differ')
    rows = [annotate(indexed[t['task_id']]) for t in tasks]
    result = summarize(rows)
    result.update(status='completed', diagnostics=diagnostics(rows), run=deepcopy(summary['run']))
    result['run'].update(tasks_sha256=digest(tasks), selected_task_ids=[t['task_id'] for t in tasks])
    result['run'].pop('elapsed_seconds', None)  # Full-run duration is not a subset measurement.
    result['provenance'] = dict(kind='reused_cpu_baseline_subset', source='results/env_v1_baseline.jsonl',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_full_run_elapsed_seconds=summary['run']['elapsed_seconds'])
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (destination/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (destination/'run.log').write_text('OFFLINE REUSE ONLY: 15 existing CPU episodes; no new inference.\nSource: results/env_v1_baseline.jsonl\n')
    return result


def write_report(output_root, contract, tasks, report, dtype=None):
    lines = ['# Model Scale Diagnostic', '',
        '固定 15 题，L1/L2/L3 各 5 题；任务 ID、权重 revision 和行为源码哈希保存在 configs/env_v1_diagnostic_tasks.json 与 configs/env_v1_model_scale.json。', '',
        '| Model | Overall | L1 | L2 | L3 | Max-tool-call rate | Repeated actions | Avg cost | Deep search calls | Status / runtime |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    completed = []
    for model, spec in contract['models'].items():
        path = output_root/spec['directory']/'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {'status': 'not_run'}
        if summary.get('status') == 'not_run':
            lines.append(f'| {model} | — | — | — | — | — | — | — | — | 待云端运行 |')
            continue
        validate_run(summary, contract, tasks, model)
        rows = [json.loads(line) for line in (path.parent/'results.jsonl').read_text().splitlines()]
        if [r['task_id'] for r in rows] != [t['task_id'] for t in tasks]:
            raise ValueError('Incomplete or reordered model output')
        recomputed = summarize(rows)
        if any(summary[k] != value for k, value in recomputed.items()) or summary['diagnostics'] != diagnostics(rows):
            raise ValueError('Summary does not match saved episodes')
        d, rates = summary['diagnostics'], summary['success_by_difficulty']
        status = '已复用历史结果' if summary.get('provenance') else '新推理'
        runtime = summary['run']['runtime']
        lines.append(f'| {model} | {summary["overall_success"]:.1%} | {rates["L1"]:.1%} | {rates["L2"]:.1%} | {rates["L3"]:.1%} | {d["max_tool_call_rate"]:.1%} | {d["repeated_identical_action_count"]} | {summary["average_tool_cost"]:.3f} | {d["deep_search_calls"]} | {status}; {runtime["device"]}/{runtime["dtype"]} |')
        completed.append(summary)
    lines += ['', 'Repeated actions = 每条轨迹内相邻、完全相同的已执行 tool name + arguments 的额外调用数之和；JSON 键顺序不影响判定。例如 A,A,A 计 2 次，A,B,A 计 0 次。包含执行后失败的调用，不含解析失败或预算阻止的提案。', '',
        '`answer_ready_but_continued` 统一为 null：本阶段不严格推断 observation 是否足以构造答案，仅使用上述连续重复指标；null 不是 false。', '',
        '统一 failure_rate=0、do_sample=false、max_tool_calls=8、max_steps=9、max_new_tokens=256、max_context_tokens=16384、seed=2026；云端默认 bfloat16，表中的 float32 仅为历史参考，不能替代本轮 bfloat16 结果。没有更改环境、task、tool schema、system prompt、evaluator 或 reward。', '']
    comparable = len(completed) == 3 and len({(s['run']['runtime']['device'], s['run']['runtime']['dtype'], s['run']['runtime'].get('torch_version'), s['run']['runtime'].get('transformers_version')) for s in completed}) == 1
    comparable = comparable and (dtype is None or all(s['run']['runtime']['dtype'] == dtype for s in completed))
    lines += ['## 当前结论', '']
    if not comparable:
        lines += ['证据尚不足以区分 1.5B 的 agent capability floor 与环境/protocol 问题。待三个模型在相同云端设备、精度和依赖版本下完成后再对照；CPU 归档不能当作完全相同 CUDA 配置的结果。', '']
    else:
        lines += ['三个模型已在相同运行配置下完成。观察成功率、预算耗尽与重复调用是否随规模改善；改善支持能力限制解释，共同失败则提示协议/环境问题，但单个固定小样本不能证明因果。', '']
    lines += ['## 云端复现', '',
        '实际 Git 根目录是 experiment/。下面命令从 clone 后的仓库根目录执行；若使用当前 ToolRL 工作区，先 cd experiment。详见 [MODEL_SCALE_RUNBOOK.md](MODEL_SCALE_RUNBOOK.md)。', '',
        '```bash', 'python -m pip install -r requirements-inference.lock.txt -r requirements-test.txt', 'pytest -q',
        'python scripts/run_model_scale.py --download-only',
        'python scripts/run_model_scale.py --device cuda --dtype bf16 --output-root results/model_scale_cuda_bfloat16',
        '```', '',
        '不同输出根目录用于保留已归档 CPU 结果；每个模型目录仍包含 results.jsonl、summary.json、run.log。正式对照统一在 CUDA/bfloat16 上重新跑三个模型，无逐模型调参。', '']
    report.write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true', help='Reuse CPU archive and mark other models not run; no model loading')
    mode.add_argument('--download-only', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dtype', type=normalize_dtype, choices=['float32', 'bfloat16'], help='bf16 alias accepted; default CUDA=bfloat16, CPU=float32')
    p.add_argument('--output-root', type=Path)
    args = p.parse_args()
    try:
        dtype = resolve_dtype(args.device, args.dtype)
    except ValueError as exc:
        p.error(str(exc))
    if args.output_root is None:
        args.output_root = ROOT/'results'/('model_scale' if args.prepare else f'model_scale_{args.device}_{dtype}')
    contract, tasks = load_contract()
    if args.download_only:
        from huggingface_hub import snapshot_download
        for model, spec in contract['models'].items():
            snapshot_download(model, revision=spec['revision'], cache_dir=str(ROOT.parent/'.hf-cache/hub'),
                              allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken'])
        return
    if not args.report_only:
        for model, spec in contract['models'].items():
            directory = args.output_root/spec['directory']
            summary_path = directory/'summary.json'
            if summary_path.exists():
                existing = json.loads(summary_path.read_text())
                if existing.get('status') != 'not_run':
                    validate_run(existing, contract, tasks, model, device=None if args.prepare else args.device, dtype=None if args.prepare else dtype)
                    continue
            if (directory/'results.jsonl').exists() and (directory/'results.jsonl').stat().st_size:
                raise ValueError('Partial output exists; preserve it and choose a fresh output root')
            directory.mkdir(parents=True, exist_ok=True)
            if args.prepare:
                if model.endswith('1.5B-Instruct'):
                    reuse_baseline(directory, contract, tasks)
                else:
                    (directory/'results.jsonl').write_text('')
                    summary_path.write_text(json.dumps(dict(status='not_run', model=model, revision=spec['revision'], task_ids=[t['task_id'] for t in tasks]), indent=2)+'\n')
                    (directory/'run.log').write_text('NOT RUN: cloud inference pending. No performance measurements.\n')
                continue
            command = [sys.executable, str(ROOT/'scripts/run_env_v1.py'), '--model', model,
                       '--device', args.device, '--dtype', dtype, '--task-ids-file', str(ROOT/'configs/env_v1_diagnostic_tasks.json'),
                       '--failure-rate', '0', '--output', str(directory/'results.jsonl')]
            environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            with (directory/'run.log').open('w') as log:
                log.write(json.dumps(dict(command=command))+'\n'); log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
            validate_run(json.loads(summary_path.read_text()), contract, tasks, model, args.device, dtype)
    write_report(args.output_root, contract, tasks, ROOT/'MODEL_SCALE_DIAGNOSTIC.md', dtype=None if args.prepare else dtype)


if __name__ == '__main__':
    main()
