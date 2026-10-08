"""Read-only paired NF4 inference with exact generated-token capture."""
from contextlib import nullcontext
from copy import deepcopy
import torch
from src.policies.huggingface import HuggingFacePolicy
from src.policies.schema import build_messages
from src.training.grpo_v1.policy import TrainingPolicy, load_policy
from src.env_v1 import Environment
from src.env_v1.tasks import initial_state
from src.env_v1.evaluation import evaluate
from src.rollouts import rollout, RolloutConfig
from src.tasks import public_task
from .audit import file_hash


class EvaluationPolicy(TrainingPolicy):
    def act(self, state):
        self.model.eval()
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
        with context:
            output = HuggingFacePolicy.act(self, state)
        self.segments.append(dict(messages=deepcopy(build_messages(state)), token_ids=list(output.token_ids), raw_output=output.raw_output))
        return output


def load_evaluation_policy(snapshot, checkpoint, config):
    from transformers import AutoConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with bf16 required; no CPU/model-size fallback')
    import bitsandbytes  # noqa: F401
    import peft  # noqa: F401
    actual = AutoConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    # Bind the local base to the exact recorded snapshot, not merely a compatible shape.
    if snapshot.name != config['revision']:
        raise ValueError('Snapshot directory must have the recorded HF revision name')
    if actual.model_type != 'qwen2': raise ValueError('Unexpected model architecture')
    loaded = load_policy(snapshot,config,checkpoint=checkpoint)
    model = loaded.model
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    saved = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cpu')
    restored = get_peft_model_state_dict(model)
    if saved.keys() != restored.keys() or any(not torch.equal(value.cpu(), saved[key].to(value.dtype)) for key, value in restored.items()):
        raise ValueError('Loaded adapter tensors do not exactly match checkpoint')
    del saved, restored
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    inference_config = deepcopy(config)
    inference_config['sampling']['do_sample'] = False
    policy = EvaluationPolicy(model,loaded.tokenizer,inference_config)
    policy.runtime_metadata.update(loaded.runtime_metadata, gradient_checkpointing=False,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
        snapshot_config_sha256=file_hash(snapshot/'config.json'))
    return policy


def collect(items, policy, config, failure_rate=0, failure_seed=2026):
    rows=[]
    for item in items:
        env=Environment(initial_state(item['kb'],item['task']), max_tool_calls=config['rollout']['max_tool_calls'],
            failure_rate=failure_rate, failure_type='timeout', seed=failure_seed)
        trajectory=rollout(public_task(item['task']),policy,env,RolloutConfig(**config['rollout']))
        if trajectory.termination_reason=='policy_error':
            raise RuntimeError(str(trajectory.metadata))
        row=evaluate(item['task'],env)
        row.update(generation_segments=deepcopy(policy.last_session.segments),
            generated_tokens=sum(len(s['token_ids']) for s in policy.last_session.segments),
            failure_rate=failure_rate, failure_seed=failure_seed, failure_mechanism='existing pre-execution timeout')
        rows.append(row)
    return rows


def fresh_pair(items,policy,config,failure_rate=0,failure_seed=2026):
    with policy.model.disable_adapter():
        before=collect(items,policy,config,failure_rate,failure_seed)
    # PEFT's adapter context may restore requires_grad flags; inference stays frozen.
    for parameter in policy.model.parameters(): parameter.requires_grad_(False)
    after=collect(items,policy,config,failure_rate,failure_seed)
    return before,after
