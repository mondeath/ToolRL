"""Local causal-LM inference only, no judge, optimizer, or training dependencies."""
import copy
import hashlib
from . import PolicyOutput
from .schema import build_messages


class ContextLimitError(RuntimeError):
    pass


class HuggingFacePolicy:
    def __init__(self, config):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config = copy.deepcopy(config)
        self.model_name = config['model_name']
        requested = config['device']
        if requested == 'auto': requested = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested.startswith('cuda') and not torch.cuda.is_available():
            if not config.get('cpu_fallback', True): raise RuntimeError('CUDA unavailable')
            requested = 'cpu'
        if requested != 'cpu' and not requested.startswith('cuda'): raise ValueError('device must be auto, cpu, or cuda[:index]')
        self.device = torch.device(requested)
        dtype_name = config['dtype']
        if self.device.type == 'cpu': dtype_name = 'float32'
        elif dtype_name == 'auto': dtype_name = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        if dtype_name not in ('float32', 'float16', 'bfloat16'): raise ValueError('Unsupported dtype')
        if config.get('cpu_threads'): torch.set_num_threads(config['cpu_threads'])
        self.sampling_config = {key: config[key] for key in ('temperature', 'top_p', 'max_new_tokens', 'do_sample')}
        if config['temperature'] <= 0 or not 0 < config['top_p'] <= 1 or config['max_new_tokens'] < 1:
            raise ValueError('Invalid sampling configuration')
        kwargs = {key: config[key] for key in ('revision', 'cache_dir', 'local_files_only') if key in config}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False, **kwargs)
        if not self.tokenizer.chat_template: raise ValueError('An instruct tokenizer with chat_template is required')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=getattr(torch, dtype_name), trust_remote_code=False, **kwargs
        ).to(self.device).eval()
        self.runtime_metadata = {'device': str(self.device), 'dtype': dtype_name,
                                 'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                                 'resolved_revision': getattr(self.model.config, '_commit_hash', None)}
        self.seed, self.call_index = 0, 0

    def new_session(self, seed):
        session = copy.copy(self)  # Weights/tokenizer shared, no conversation/KV cache shared.
        session.seed, session.call_index = seed, 0
        return session

    def act(self, state):
        import torch
        prompt = self.tokenizer.apply_chat_template(build_messages(state), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        limit = min(self.config['max_context_tokens'], getattr(self.model.config, 'max_position_embeddings', self.config['max_context_tokens']))
        if inputs['input_ids'].shape[-1] + self.config['max_new_tokens'] > limit:
            raise ContextLimitError('Prompt plus generation budget exceeds context limit; history was not silently dropped')
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.call_index}'.encode()).digest()[:8], 'big') % (2**63-1)
        self.call_index += 1
        cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        generation = dict(max_new_tokens=self.config['max_new_tokens'], do_sample=self.config['do_sample'],
                          pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                          use_cache=True)
        if generation['do_sample']: generation.update(temperature=self.config['temperature'], top_p=self.config['top_p'])
        # Sequential runner: save/restore global RNG so independent rollouts cannot influence one another.
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]): torch.cuda.manual_seed(seed)
            output = self.model.generate(**inputs, **generation)
        tokens = output[0, inputs['input_ids'].shape[-1]:].tolist()
        return PolicyOutput(self.tokenizer.decode(tokens, skip_special_tokens=True), tokens)
