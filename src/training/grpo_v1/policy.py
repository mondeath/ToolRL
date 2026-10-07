"""QLoRA policy with exact generation capture, using the frozen HF act method."""
from copy import deepcopy
from contextlib import nullcontext
import math
import torch

from src.policies.huggingface import HuggingFacePolicy
from src.policies.schema import build_messages
from .loss import make_segment, generated_logprobs


def assert_adapter_only(model):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('.lora_A.' not in name and '.lora_B.' not in name for name, _ in trainable):
        raise ValueError('Optimizer may train only LoRA A/B; base must be frozen')
    return trainable


def parameter_counts(model):
    trainable = sum(p.numel() for _, p in assert_adapter_only(model))
    total, storage = 0, 0
    for p in model.parameters():
        # Packed uint8 numel is not the logical number of 4-bit parameters.
        if p.__class__.__name__ == 'Params4bit':
            if getattr(p, 'quant_state', None) is None:
                raise ValueError('4-bit parameter is not quantized')
            total += math.prod(p.quant_state.shape)
        else:
            total += p.numel()
        storage += p.numel() * p.element_size()
    return dict(total_params=total, trainable_params=trainable, trainable_ratio=trainable / total,
                parameter_storage_bytes=storage)


class TrainingPolicy(HuggingFacePolicy):
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer = model, tokenizer
        self.config = deepcopy(config['sampling'])
        self.model_name = config['model_name']
        self.device = next(model.parameters()).device
        self.sampling_config = {key: self.config[key] for key in
            ('do_sample', 'temperature', 'top_p', 'top_k', 'max_new_tokens')}
        self.runtime_metadata = dict(device=str(self.device), dtype=str(next(model.parameters()).dtype),
            quantization='NF4' if getattr(model, 'is_loaded_in_4bit', False) else 'synthetic_or_unquantized',
            compute_dtype='bfloat16' if self.device.type == 'cuda' else 'float32',
            logprob_space='raw_model_softmax', beta=0.0)
        self.seed, self.call_index = 0, 0
        self.capture = True
        self.last_session = None
        self.segments = []
        # HF act forwards temperature/top_p; set the remaining explicit sampler
        # parameter on the shared generation config. Greedy evaluation ignores it.
        if getattr(model, 'generation_config', None) is not None:
            model.generation_config.top_k = self.config['top_k']

    def new_session(self, seed):
        session = super().new_session(seed)
        session.segments = []
        self.last_session = session
        return session

    def act(self, state):
        self.model.eval()
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
        with context:
            output = super().act(state)  # Identical template, parser input, seed and context checks.
        if self.capture:
            messages = build_messages(state)
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = self.tokenizer(text, add_special_tokens=False)['input_ids']
            segment = make_segment(deepcopy(messages), prompt, output.token_ids, output.raw_output,
                                   generation_index=self.call_index - 1, session_seed=self.seed)
            with torch.no_grad():
                segment['old_logprobs'] = generated_logprobs(self.model, segment).cpu().tolist()
            self.segments.append(segment)
        return output


def load_policy(snapshot, config, checkpoint=None):
    """Called only by explicit --train / --eval, never by static/preflight."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    quantization = dict(config['quantization'], bnb_4bit_compute_dtype=torch.bfloat16)
    torch.manual_seed(config['seed'])
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(**quantization), device_map={'': 0}, attn_implementation='sdpa')
    if not getattr(model, 'is_loaded_in_4bit', False):
        raise ValueError('Refuse non-QLoRA model')
    counts = {name: sum(path.endswith('.' + name) for path, module in model.named_modules())
              for name in config['lora']['target_modules']}
    if any(count != model.config.num_hidden_layers for count in counts.values()):
        raise ValueError(f'Unexpected Qwen projection names/counts: {counts}')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    # prepare_model_for_kbit_training casts non-quantized tensors to fp32.
    # Restore frozen embeddings/head to bf16 to keep hidden states and KV cache
    # in the intended compute dtype; leave normalization weights in fp32.
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module.weight.__class__.__name__ == 'Params4bit':
            raise ValueError('Expected unquantized Qwen embeddings/lm_head')
        module.to(dtype=torch.bfloat16)
    if checkpoint:
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', **config['lora']))
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    assert_adapter_only(model)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError('Instruct tokenizer with chat_template required')
    policy = TrainingPolicy(model, tokenizer, config)
    policy.runtime_metadata.update(parameter_counts(model), target_module_counts=counts,
                                   revision=config['revision'], gradient_checkpointing=model.is_gradient_checkpointing)
    if not model.is_gradient_checkpointing:
        raise ValueError('Gradient checkpointing was not activated')
    return policy
