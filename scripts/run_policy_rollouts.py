"""Run actual local model inference; no mock-policy fallback or reward computation."""
from _common import ROOT, json
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import time
from src.environment import ToolEnvironment
from src.evaluation.policy import evaluate_policy, write_report
from src.policies.huggingface import HuggingFacePolicy
from src.policies import AgentState
from src.policies.schema import SYSTEM_PROMPT, ACTION_SCHEMA, build_messages
from src.rollouts import rollout, RolloutConfig


def select_tasks(public, evaluation, count):
    if not 1 <= count <= len(public): raise ValueError('task_count out of range')
    # Stratification uses only task_id/type; never sends evaluation records into the runner.
    kinds = {t['task_id']: t['task_type'] for t in evaluation}
    buckets = defaultdict(list)
    for task in public:
        if set(task) != {'task_id', 'question'}: raise ValueError('Public tasks contain forbidden fields')
        buckets[kinds[task['task_id']]].append(task)
    selected = []
    while len(selected) < count:
        for kind in sorted(buckets):
            if buckets[kind] and len(selected) < count: selected.append(buckets[kind].pop(0))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/policy.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['group_size'] < 1: raise ValueError('group_size must be positive')
    public_path, evaluation_path, kb_path = [ROOT/'data'/name for name in ('tasks.public.json', 'tasks.evaluation.json', 'kb.json')]
    public = json.loads(public_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    selected = select_tasks(public, evaluation, config['task_count'])
    kb = json.loads(kb_path.read_text())
    rollout_config = RolloutConfig(**config['rollout'])
    model_config = dict(config['policy'])
    if model_config.get('cache_dir'): model_config['cache_dir'] = str((ROOT/model_config['cache_dir']).resolve())
    policy = HuggingFacePolicy(model_config)
    if policy.runtime_metadata['device'] == 'cpu' and len(selected)*config['group_size'] > config.get('cpu_trajectory_limit', 8):
        raise ValueError('CPU is reserved for small smoke tests; use policy_cpu_smoke.json or run the full config on CUDA')
    output_dir = ROOT/config.get('output_dir', 'data')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'policy_prompt_example.json').write_text(json.dumps(
        build_messages(AgentState(selected[0]['question'], [], ACTION_SCHEMA)),
        ensure_ascii=False, indent=2) + '\n')
    trajectories = []
    started = time.time()
    # Persist every completed rollout, including invalid/truncated outcomes.
    with (output_dir/'policy_rollouts.jsonl').open('w', encoding='utf-8') as stream:
        for task in selected:
            for i in range(config['group_size']):
                env = ToolEnvironment(kb, top_k=config['environment']['top_k'], max_tool_calls=max(1, rollout_config.max_tool_calls))
                trajectory = rollout(task, policy, env, rollout_config, rollout_id=i)
                trajectories.append(trajectory)
                stream.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                print(f"{len(trajectories)}/{len(selected)*config['group_size']} {task['task_id']} rollout={i} termination={trajectory.termination_reason} tools={trajectory.tool_sequence}", flush=True)
    summary = evaluate_policy(evaluation, trajectories)
    summary['run'] = {'config': config, 'runtime': policy.runtime_metadata, 'elapsed_seconds': time.time()-started,
                      'task_ids': [t['task_id'] for t in selected],
                      'kb_sha256': hashlib.sha256(kb_path.read_bytes()).hexdigest(),
                      'public_tasks_sha256': hashlib.sha256(public_path.read_bytes()).hexdigest(),
                      'evaluation_tasks_sha256': hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
                      'prompt_sha256': hashlib.sha256((SYSTEM_PROMPT+json.dumps(ACTION_SCHEMA, sort_keys=True)).encode()).hexdigest()}
    (output_dir/'policy_rollout_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    write_report(ROOT/config.get('report_path', 'reports/base_policy_analysis.md'), summary, trajectories)
    print(json.dumps(summary['aggregate'], indent=2), flush=True)


if __name__ == '__main__': main()
