"""40-task real Qwen baseline. Independent v1 outputs; flush after every episode."""
from _common import ROOT
import argparse
import hashlib
import json
from pathlib import Path
import time
from src.env_v1 import Environment, generate_kb
from src.env_v1.tasks import generate_tasks, initial_state
from src.env_v1.evaluation import evaluate, summarize
from src.evaluation.model_scale import select_tasks, annotate, diagnostics, normalize_dtype, resolve_dtype
from src.env_v1.protocol import ACTION_SCHEMA, SYSTEM_PROMPT
from src.tasks import public_task
from src.rollouts import rollout, RolloutConfig
from src.policies.huggingface import HuggingFacePolicy
from src.policies import AgentState
from src.policies.schema import build_messages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failure-rate', type=float, default=0)
    p.add_argument('--output', type=Path)
    p.add_argument('--model', help='One of the three pinned Qwen2.5 diagnostic models')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Common execution device')
    p.add_argument('--dtype', type=normalize_dtype, choices=['float32', 'bfloat16'], help='bf16 alias accepted; default CUDA=bfloat16, CPU=float32')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--task-ids', nargs='+')
    group.add_argument('--task-ids-file', type=Path)
    p.add_argument('--task-count', type=int, default=40)
    args = p.parse_args()
    if not 1 <= args.task_count <= 40: p.error('task-count must be 1..40')
    if not 0 <= args.failure_rate <= 1: p.error('failure-rate must be 0..1')
    try:
        dtype = resolve_dtype(args.device, args.dtype)
    except ValueError as exc:
        p.error(str(exc))
    kb = generate_kb()
    ids = args.task_ids
    if args.task_ids_file:
        ids = json.loads(args.task_ids_file.read_text())['task_ids']
    try:
        tasks = select_tasks(generate_tasks(kb), ids, args.task_count)
    except ValueError as exc:
        p.error(str(exc))
    models = json.loads((ROOT/'configs/env_v1_model_scale.json').read_text())['models']
    if args.model and args.model not in models:
        p.error('model must be one of: ' + ', '.join(models))
    if args.output is None:
        if args.model or args.device != 'cpu':
            model = args.model or 'Qwen/Qwen2.5-1.5B-Instruct'
            args.output = ROOT/'results'/f'model_scale_{args.device}_{dtype}'/models[model]['directory']/'results.jsonl'
        else:
            args.output = ROOT.parent/'results/env_v1_baseline.jsonl'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/policy_cpu_smoke.json').read_text())['policy']
    config.update(cache_dir=str((ROOT/config['cache_dir']).resolve()), do_sample=False, max_new_tokens=256, max_context_tokens=16384)
    config['device'] = args.device
    config['dtype'] = dtype
    if args.device == 'cuda':
        config['cpu_fallback'] = False
    if args.model:
        config.update(model_name=args.model, revision=models[args.model]['revision'])
    rc = RolloutConfig(max_steps=9, max_tool_calls=8, seed=2026)
    # Persist frozen inputs alongside run, including explicit final-state assertions.
    args.output.with_suffix('.tasks.json').write_text(json.dumps(tasks, indent=2)+'\n')
    args.output.with_suffix('.kb.json').write_text(json.dumps(kb, indent=2)+'\n')
    args.output.with_suffix('.prompt.json').write_text(json.dumps(build_messages(AgentState(tasks[0]['question'], [], ACTION_SCHEMA)), indent=2)+'\n')
    started = time.time()
    policy = HuggingFacePolicy(config)
    rows = []
    with args.output.open('w') as stream:
        for task in tasks:
            env = Environment(initial_state(kb, task), failure_rate=args.failure_rate)
            rollout(public_task(task), policy, env, rc)
            row = annotate(evaluate(task, env))
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            print(f'{len(rows)}/{len(tasks)} {task["task_id"]} success={row["success"]} cost={row["tool_cost"]} termination={env.trajectory.termination_reason}', flush=True)
    summary = summarize(rows)
    summary['diagnostics'] = diagnostics(rows)
    summary['run'] = dict(model_config=config, runtime=policy.runtime_metadata, failure_rate=args.failure_rate,
        selected_task_ids=[t['task_id'] for t in tasks],
        rollout_config=rc.__dict__, elapsed_seconds=time.time()-started, kb_seed=42,
        kb_sha256=hashlib.sha256(json.dumps(kb, sort_keys=True).encode()).hexdigest(),
        tasks_sha256=hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
        prompt_sha256=hashlib.sha256((SYSTEM_PROMPT+json.dumps(ACTION_SCHEMA, sort_keys=True)).encode()).hexdigest())
    summary_path = args.output.with_name('summary.json') if args.output.name == 'results.jsonl' else args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
