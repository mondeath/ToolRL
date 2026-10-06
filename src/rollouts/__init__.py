"""Rollout execution only: this module never imports task evaluation or rewards."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from src.policies import AgentState
from src.policies.schema import ACTION_SCHEMA
from src.policies.parser import parse_action
from src.policies.normalizer import NormalizationResult, normalize_protocol_output
from src.policies.huggingface import ContextLimitError


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 8
    max_tool_calls: int = 6
    max_parse_retries: int = 1
    seed: int = 2026

    def __post_init__(self):
        if self.max_steps < 1 or self.max_tool_calls < 0 or self.max_parse_retries not in (0, 1):
            raise ValueError('Require positive max_steps, nonnegative tool budget, and 0 or 1 repair retry')


def sampling_seed(base_seed, task_id, rollout_id):
    return int.from_bytes(hashlib.sha256(f'{base_seed}:{task_id}:{rollout_id}'.encode()).digest()[:8], 'big') % (2**63-1)


def rollout(task, policy, env, config, rollout_id=0):
    """task must be public_task. Every call creates a fresh policy session and resets env."""
    env.reset(task)
    t = env.trajectory
    seed = sampling_seed(config.seed, task['task_id'], rollout_id)
    session = policy.new_session(seed)
    t.policy_model = session.model_name
    t.sampling_config = deepcopy(session.sampling_config)
    t.sampling_config['seed'] = seed
    t.metadata.update(rollout_id=rollout_id, rollout_config=asdict(config),
                      runtime=deepcopy(getattr(session, 'runtime_metadata', {})))
    history = []
    for step_index in range(config.max_steps):
        t.number_of_steps += 1
        parsed = None
        for retry in range(config.max_parse_retries + 1):
            state = AgentState(question=task['question'], history=deepcopy(history), action_schema=deepcopy(getattr(env, 'action_schema', ACTION_SCHEMA)))
            try:
                output = session.act(state)
            except (ContextLimitError, RuntimeError) as exc:
                t.metadata['inference_error'] = str(exc)
                t.termination_reason = 'context_limit' if isinstance(exc, ContextLimitError) else 'policy_error'
                t.truncated = True
                return t
            action_parser = getattr(env, 'parse_action', parse_action)
            strict_result = action_parser(output.raw_output)
            result = strict_result
            normalization = NormalizationResult(output.raw_output)
            if not strict_result.success:
                normalization = normalize_protocol_output(output.raw_output)
                if normalization.normalization_applied:
                    result = action_parser(normalization.normalized_output)
            normalized_success = normalization.normalization_applied and result.success
            if result.success:
                outcome = 'repair_success' if retry > 0 else ('normalized_success' if normalized_success else 'strict_success')
            else:
                outcome = 'final_invalid_action' if retry == config.max_parse_retries else 'repair_pending'
            t.raw_outputs.append(output.raw_output)
            t.parse_status.append({
                'step': step_index + 1, 'retry_count': retry, **asdict(result),
                **asdict(normalization), 'strict_success': strict_result.success,
                'strict_error': strict_result.error, 'normalized_success': normalized_success,
                'repair_success': retry > 0 and result.success, 'parse_outcome': outcome,
            })
            # Token fields intentionally remain empty until a training-aware alignment contract exists.
            if result.success:
                parsed = result.parsed_action
                break
            history.append({'raw_output': output.raw_output, 'format_feedback':
                            'Invalid action format: ' + result.error + '. Return exactly one valid action JSON object using the supplied schema.'})
        if parsed is None:
            t.steps.append({'action': 'invalid_action', 'raw_output_indices': [i for i,p in enumerate(t.parse_status) if p['step'] == step_index+1], 'cost': 0})
            t.termination_reason = 'invalid_action'
            return t
        action, arguments = parsed['action'], parsed['arguments']
        if action != 'direct_answer' and len(t.tool_calls) >= min(config.max_tool_calls, env.max_tool_calls):
            t.steps.append({'action': 'blocked_tool_call', 'proposed_action': deepcopy(parsed), 'cost': 0})
            t.termination_reason, t.truncated = 'max_tool_calls', True
            return t
        observation = env.step(action, **arguments)
        t.tool_sequence.append(action)
        if action == 'direct_answer':
            t.termination_reason = 'direct_answer'
            return t
        # Do not forward env.steps, tool_calls, total_cost, or any metadata to the policy.
        history.append({'action': deepcopy(parsed), 'observation': deepcopy(observation)})
    t.termination_reason, t.truncated = 'max_steps', True
    return t


def rollout_group(task, policy, env_factory, config=None, G=4):
    """Factory must construct the same initial KB/config for each independent environment."""
    if G < 1: raise ValueError('G must be positive')
    config = config or RolloutConfig()
    return [rollout(task, policy, env_factory(), config, rollout_id=i) for i in range(G)]
