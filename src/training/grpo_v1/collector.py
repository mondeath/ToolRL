"""Reuse frozen rollout and evaluator, adding a separate training record."""
from copy import deepcopy

from src.env_v1 import Environment
from src.env_v1.tasks import initial_state
from src.env_v1.evaluation import evaluate
from src.rollouts import rollout, RolloutConfig
from src.tasks import public_task
from .loss import validate_segment


def task_reward(evaluation):
    return float(evaluation['task_success'])


def collect_group(item, policy, config, group_id):
    if config['group_size'] != 4 or not policy.config['do_sample']:
        raise ValueError('Training collector requires G=4 and sampling')
    task, kb = item['task'], item['kb']
    records = []
    policy.capture = True
    for sample_id in range(4):
        env = Environment(initial_state(kb, task), max_tool_calls=8, failure_rate=0)
        trajectory = rollout(public_task(task), policy, env, RolloutConfig(**config['rollout']), rollout_id=sample_id)
        if trajectory.termination_reason == 'policy_error':
            raise RuntimeError('Rollout infrastructure failure: ' + trajectory.metadata.get('inference_error', 'unknown'))
        segments = deepcopy(policy.last_session.segments)
        if not segments:
            raise RuntimeError('No generated tokens; cannot form a training trajectory')
        if len(segments) != len(trajectory.raw_outputs):
            raise ValueError('Missing generation capture (including repair)')
        for segment in segments:
            validate_segment(segment)
            if len(segment['old_logprobs']) != len(segment['generated_token_ids']):
                raise ValueError('Missing old token logprobs')
        evaluation = evaluate(task, env)
        records.append(dict(task_id=task['task_id'], seed=item['seed'], difficulty=task['difficulty'],
            group_id=group_id, sample_id=sample_id, session_seed=trajectory.sampling_config['seed'],
            messages=[segment['messages'] for segment in segments], actions=deepcopy(trajectory.steps),
            observations=deepcopy(trajectory.observations), termination_reason=trajectory.termination_reason,
            success=evaluation['task_success'], reward=task_reward(evaluation), tool_cost=evaluation['tool_cost'],
            rule_violations=evaluation['rule_violations'], trajectory_length=evaluation['trajectory_length'],
            generated_token_count=sum(len(s['generated_token_ids']) for s in segments),
            segments=segments, trajectory=trajectory.to_dict()))
    return records
