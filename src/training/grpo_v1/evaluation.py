"""Greedy diagnostics via the original evaluator, without reward changes."""
from contextlib import contextmanager
import json

from src.env_v1 import Environment
from src.env_v1.tasks import initial_state
from src.env_v1.evaluation import evaluate, summarize
from src.evaluation.model_scale import diagnostics
from src.rollouts import rollout, RolloutConfig
from src.tasks import public_task


@contextmanager
def greedy_mode(policy):
    previous = policy.config['do_sample'], policy.sampling_config['do_sample'], policy.capture
    policy.config['do_sample'] = policy.sampling_config['do_sample'] = False
    policy.capture = False
    try:
        yield
    finally:
        policy.config['do_sample'], policy.sampling_config['do_sample'], policy.capture = previous


def metrics(rows):
    summary = summarize(rows)
    summary.update(diagnostics(rows))
    summary.update(invalid_action_rate=sum(r['trajectory']['termination_reason'] == 'invalid_action' for r in rows) / len(rows),
        success_zero_violation=sum(r['task_success'] and not r['rule_violation_count'] for r in rows) / len(rows))
    parses = [p for row in rows for p in row['trajectory']['parse_status']]
    summary['parsed_action_rate'] = sum(p['success'] for p in parses) / len(parses) if parses else 0.0
    summary['has_legal_tool_actions'] = any(row['tool_calls'] > 0 for row in rows)
    return summary


def greedy_eval(items, policy, config, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    with greedy_mode(policy), (directory / 'results.jsonl').open('w') as stream:
        for item in items:
            env = Environment(initial_state(item['kb'], item['task']), failure_rate=0)
            trajectory = rollout(public_task(item['task']), policy, env, RolloutConfig(**config['rollout']))
            if trajectory.termination_reason == 'policy_error':
                raise RuntimeError('Greedy evaluation inference failure: ' + str(trajectory.metadata))
            row = evaluate(item['task'], env)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
    result = metrics(rows)
    result.update(do_sample=False, task_ids=[x['task']['task_id'] for x in items])
    (directory / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
