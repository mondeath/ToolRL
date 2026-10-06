"""Run the unassisted 8x1 baseline on CUDA only; no training or model download."""
from _common import ROOT, json
import argparse
import hashlib
from pathlib import Path
import platform
import time

from run_policy_rollouts import select_tasks
from src.environment import ToolEnvironment
from src.evaluation.policy import evaluate_policy, write_report
from src.policies.huggingface import HuggingFacePolicy
from src.policies.schema import SYSTEM_PROMPT, ACTION_SCHEMA
from src.rewards import CostAwareReward, MockAIReward
from src.rollouts import rollout, RolloutConfig


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_baseline(output_dir, cache_dir, model_name=None, revision=None):
    if (model_name is None) != (revision is None):
        raise ValueError("model_name and revision must be supplied together")
    output_dir = Path(output_dir).expanduser().resolve()
    config_path = ROOT / 'configs/policy.json'
    reward_config_path = ROOT / 'configs/default.json'
    config = json.loads(config_path.read_text())
    if model_name is not None:
        config['policy'].update(model_name=model_name, revision=revision)
    config['policy'].update(device='cuda:0', cpu_fallback=False, dtype='auto',
                            do_sample=False, local_files_only=True,
                            cache_dir=str(Path(cache_dir).expanduser().resolve()))
    config.update(task_count=8, group_size=1,
                  rollout=dict(max_steps=4, max_tool_calls=3, max_parse_retries=1, seed=2026),
                  output_dir=str(output_dir), report_path=str(output_dir / 'report.md'))
    config.pop('cpu_trajectory_limit', None)
    paths = {name: ROOT / 'data' / filename for name, filename in (
        ('public_tasks', 'tasks.public.json'), ('evaluation_tasks', 'tasks.evaluation.json'), ('kb', 'kb.json'))}
    public, evaluation, kb = [json.loads(paths[name].read_text()) for name in ('public_tasks', 'evaluation_tasks', 'kb')]
    selected = select_tasks(public, evaluation, config['task_count'])
    task_ids = [task['task_id'] for task in selected]
    if len(set(task_ids)) != 8:
        raise ValueError('Baseline requires eight distinct task IDs')
    cost_lambda = json.loads(reward_config_path.read_text())['cost_lambda']
    reward = CostAwareReward(MockAIReward(), cost_lambda=cost_lambda)
    run = {
        'experiment': 'unassisted GPU baseline; no training', 'config': config,
        'task_ids': task_ids, 'selection': 'select_tasks: sorted task-type round robin, public file order within type',
        'python_version': platform.python_version(),
        'offline_reward': {'scorer': 'CostAwareReward(MockAIReward)', 'cost_lambda': cost_lambda,
                           'note': 'Evidence-agreement stub; not the final research reward model. Separate from policy evaluation.'},
        'prompt_sha256': hashlib.sha256((SYSTEM_PROMPT + json.dumps(ACTION_SCHEMA, sort_keys=True)).encode()).hexdigest(),
    }
    for name, path in {**paths, 'source_policy_config': config_path, 'source_reward_config': reward_config_path}.items():
        run[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Exclusive creation reserves the experiment before loading any model weights.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'experiment_config.json', config)
    write_json(output_dir / 'selected_task_ids.json', task_ids)
    write_json(output_dir / 'runtime_metadata.json', run)
    started = time.time()
    policy = HuggingFacePolicy(config['policy'])
    policy.runtime_metadata.update(
        model_name=config['policy']['model_name'], revision=config['policy']['revision'])
    run['runtime'] = policy.runtime_metadata
    write_json(output_dir / 'runtime_metadata.json', run)
    rollout_config = RolloutConfig(**config['rollout'])
    trajectories = []
    with (output_dir / 'policy_rollouts.jsonl').open('x', encoding='utf-8') as stream, \
         (output_dir / 'mock_ai_rewards.jsonl').open('x', encoding='utf-8') as rewards:
        for task in selected:
            env = ToolEnvironment(kb, top_k=config['environment']['top_k'], max_tool_calls=rollout_config.max_tool_calls)
            trajectory = rollout(task, policy, env, rollout_config, rollout_id=0)
            trajectories.append(trajectory)
            stream.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            rewards.write(json.dumps({'task_id': task['task_id'], 'rollout_id': 0,
                                      'scorer': 'CostAwareReward(MockAIReward)',
                                      'cost_lambda': cost_lambda, **reward.breakdown(trajectory)},
                                     ensure_ascii=False, allow_nan=False) + '\n')
            rewards.flush()
            print(f"{len(trajectories)}/8 {task['task_id']} termination={trajectory.termination_reason}", flush=True)
    run['elapsed_seconds'] = time.time() - started
    summary = evaluate_policy(evaluation, trajectories)
    summary['run'] = run
    write_json(output_dir / 'runtime_metadata.json', run)
    write_json(output_dir / 'policy_rollout_summary.json', summary)
    write_report(output_dir / 'report.md', summary, trajectories)
    aggregate = summary['aggregate']
    normal = sum(row['termination_reason'] == 'direct_answer' and not row['truncated'] for row in summary['task_level'])
    print(f"Tasks: {len(selected)} | Accuracy: {aggregate['normalized_accuracy']:.3f} | "
          f"Average cost: {aggregate['average_cost']:.3f} | Normal completions: {normal} | "
          f"Invalid-action terminations: {aggregate['termination_counts'].get('invalid_action', 0)} | "
          f"Tool errors: {aggregate['tool_execution_errors']} | Output: {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', help='Override policy model; requires --revision. Default: configs/policy.json.')
    parser.add_argument('--revision', help='Pinned model revision; requires --model-name. Default: configs/policy.json.')
    parser.add_argument('--cache-dir', type=Path, default=Path('/root/shared-nvme/hf-cache'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/gpu_baseline_8x1',
                        help='Must not already exist; relative paths resolve from the working directory.')
    args = parser.parse_args()
    if (args.model_name is None) != (args.revision is None):
        parser.error('--model-name and --revision must be supplied together')
    run_baseline(args.output_dir, args.cache_dir, model_name=args.model_name, revision=args.revision)


if __name__ == '__main__':
    main()
