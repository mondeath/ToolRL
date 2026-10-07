"""Read-only preflight: never load model weights, generate, or optimize."""
import importlib
import importlib.metadata
import json
from pathlib import Path

from .data import load_data


def check_snapshot(snapshot, config):
    if not snapshot:
        raise ValueError('--model-dir must point to the local frozen HF snapshot')
    path = Path(snapshot).expanduser().absolute()
    if (path.name != config['revision'] or path.parent.name != 'snapshots'
            or path.parent.parent.name != 'models--Qwen--Qwen2.5-7B-Instruct'):
        raise ValueError('Require the canonical Qwen2.5-7B HF snapshot path with frozen revision')
    model_config = json.loads((path / 'config.json').read_text())
    expected = dict(model_type='qwen2', hidden_size=3584, num_hidden_layers=28,
                    num_attention_heads=28, num_key_value_heads=4, vocab_size=152064)
    if any(model_config.get(k) != value for k, value in expected.items()):
        raise ValueError('Local config is not the frozen 7B architecture')
    index = path / 'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    if not shards:
        raise ValueError('No local model shards')
    files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'] + shards
    for name in files:
        if Path(name).name != name:
            raise ValueError('Unexpected shard path')
        with (path / name).open('rb') as stream:
            if not stream.read(1):
                raise ValueError('Empty snapshot file: ' + name)
    return dict(path=str(path), revision=config['revision'], readable_files=len(files),
                identity_basis='canonical HF cache path and architecture; not a full weight-content checksum')


def preflight(root, config, snapshot=None, require_gpu=True, data_loader=load_data):
    report = dict(status='passed', mode='gpu_preflight' if require_gpu else 'static', errors=[], checks={})
    try:
        train, diagnostic = data_loader(root, config)
        report['checks']['data'] = dict(train_tasks=len(train), train_seeds=sorted({x['seed'] for x in train}),
            diagnostic_tasks=len(diagnostic), split_overlap=False, frozen_contract=True)
    except Exception as exc:
        report['errors'].append('data: ' + str(exc))
    if require_gpu:
        report['checks']['versions'] = {}
        for module in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'peft'):
            try:
                imported = importlib.import_module(module)
                report['checks']['versions'][module] = importlib.metadata.version(module)
            except Exception as exc:
                report['errors'].append(module + ': ' + str(exc))
        versions = report['checks']['versions']
        for name, expected in [('peft', '0.17.1'), ('bitsandbytes', '0.48.1')]:
            if name in versions and versions[name] != expected:
                report['errors'].append(f'{name}: expected {expected}, found {versions[name]}')
        if 'transformers' in versions and not versions['transformers'].startswith('4.57.'):
            report['errors'].append('Use transformers 4.57.x with selective Qwen logits support')
        try:
            import torch
            available = torch.cuda.is_available()
            report['checks']['cuda_available'] = available
            report['checks']['torch_cuda_version'] = torch.version.cuda
            if not available:
                report['errors'].append('CUDA unavailable; no CPU fallback for 7B training')
            else:
                device = torch.cuda.get_device_properties(0)
                report['checks']['gpu'] = dict(name=device.name, total_memory=device.total_memory,
                    bf16_supported=torch.cuda.is_bf16_supported(), device=0)
                if not torch.cuda.is_bf16_supported():
                    report['errors'].append('CUDA device lacks bf16 support')
                if device.total_memory < 22 * 1024 ** 3:
                    report['errors'].append('This smoke requires approximately 24GiB or more GPU memory')
            from transformers import Qwen2ForCausalLM
            import inspect
            if 'logits_to_keep' not in inspect.signature(Qwen2ForCausalLM.forward).parameters:
                report['errors'].append('Transformers lacks selective Qwen logits projection')
        except Exception as exc:
            report['errors'].append('runtime: ' + str(exc))
        try:
            report['checks']['snapshot'] = check_snapshot(snapshot, config)
        except Exception as exc:
            report['errors'].append('snapshot: ' + str(exc))
    report['status'] = 'blocked' if report['errors'] else 'passed'
    report['model_loaded'] = False
    report['training_started'] = False
    return report
