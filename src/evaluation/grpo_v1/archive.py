"""Re-evaluate archived independent dev rollouts by replaying actual tool calls."""
from copy import deepcopy
import json
from src.env_v1 import Environment
from src.env_v1.tasks import initial_state
from src.env_v1.evaluation import evaluate
from src.tasks import public_task
from src.trajectories import Trajectory
from src.rollouts import sampling_seed


def replay(item, row, config):
    trajectory = row['trajectory']
    if trajectory['sampling_config']['do_sample'] or trajectory['metadata']['rollout_config'] != config['rollout']:
        raise ValueError('Archived decoding/limits mismatch')
    if trajectory['policy_model'] != config['model_name'] or trajectory['metadata']['runtime']['revision'] != config['revision']:
        raise ValueError('Archived model identity mismatch')
    expected_sampling = dict(config['sampling'], do_sample=False)
    for key in ('do_sample','temperature','top_p','top_k','max_new_tokens'):
        if trajectory['sampling_config'][key] != expected_sampling[key]:
            raise ValueError('Archived sampling mismatch: '+key)
    runtime = trajectory['metadata']['runtime']
    if runtime['quantization'] != 'NF4' or runtime['compute_dtype'] != 'bfloat16':
        raise ValueError('Archived quantization mismatch')
    if trajectory['sampling_config']['seed'] != sampling_seed(config['rollout']['seed'], item['task']['task_id'], 0):
        raise ValueError('Archived rollout seed mismatch')
    env = Environment(initial_state(item['kb'],item['task']), max_tool_calls=config['rollout']['max_tool_calls'])
    env.reset(public_task(item['task']))
    for call in trajectory['tool_calls']:
        observation = env.step(call['tool_name'], **call['arguments'])
        if observation != call['observation'] or env.trajectory.tool_calls[-1]['cost'] != call['cost']:
            raise ValueError('Archived observation/cost does not replay')
    env.trajectory = Trajectory(**deepcopy(trajectory))
    checked = evaluate(item['task'],env)
    for key in ('task_success','rule_violation_count','tool_cost','tool_calls','final_applications','failure_events'):
        if checked[key] != row[key]: raise ValueError('Original evaluator replay mismatch: '+key)
    checked.update(generated_tokens=None, source='archived independent dev eval; exact tokens unavailable', replay_verified=True)
    return checked


def archived_pair(items, run_dir, config):
    outputs=[]
    for relative in ('base_dev/results.jsonl','eval_group_240/dev/results.jsonl'):
        rows=[json.loads(line) for line in (run_dir/relative).read_text().splitlines()]
        lookup={r['task_id']:r for r in rows}
        if len(lookup)!=len(rows): raise ValueError('Duplicate archived task ID')
        outputs.append([replay(item,lookup[item['task']['task_id']],config) for item in items])
    return outputs
