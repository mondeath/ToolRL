#!/usr/bin/env python3
"""Independent evaluation. Never trains, changes old results, or saves weights."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.grpo_v1.data import read_json
from src.evaluation.grpo_v1.data import prepare_selection,load_selection,leakage_audit,subset
from src.evaluation.grpo_v1.audit import audit_run,file_hash
from src.evaluation.grpo_v1.reporting import write_json,save_pair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','audit','archive','run','robustness'])
    parser.add_argument('--run-dir',type=Path,default=ROOT/'results/grpo_v1_baseline/run_20261007_160206')
    parser.add_argument('--selection',type=Path,default=ROOT/'configs/grpo_v1_eval_tasks.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--expected-model')
    parser.add_argument('--subset',choices=['pilot','full'],default='pilot')
    parser.add_argument('--normal-run',type=Path)
    args=parser.parse_args()
    run_dir=args.run_dir.resolve()
    if args.output.exists(): raise ValueError('Output must be a NEW directory')
    if run_dir == args.output.resolve() or run_dir in args.output.resolve().parents:
        raise ValueError('Evaluation output must be outside the original training run')
    if args.mode=='prepare':
        selection=prepare_selection(ROOT,run_dir)
        if args.selection.exists():
            if read_json(args.selection)!=selection: raise ValueError('Refuse overwrite of different selection')
        else: write_json(args.selection,selection)
    selection=read_json(args.selection)
    train,items=load_selection(ROOT,run_dir,selection)
    leak=leakage_audit(train,items)
    audit=audit_run(run_dir,args.checkpoint,args.expected_model)
    config=read_json(run_dir/'config.json')
    provenance=dict(mode=args.mode,subset=args.subset,selection_sha256=selection['selection_sha256'],
        model_name=config['model_name'],revision=config['revision'],config_sha256=audit['config_sha256'],
        source_run=str(run_dir),retrospective_dev_holdout=True,completed=False)
    if args.mode in ('run','robustness'):
        if not args.expected_model: raise ValueError('--expected-model required: resolve artifact/user model discrepancy explicitly')
        if not audit['restoration_ready']: raise ValueError('; '.join(audit['blockers']))
        if not args.snapshot: raise ValueError('--snapshot required')
        checkpoint=Path(audit['checkpoint_path'])
        provenance['adapter_sha256']=file_hash(checkpoint/'adapter_model.safetensors')
        if args.mode=='robustness':
            if not args.normal_run: raise ValueError('Completed full normal run required first')
            normal=read_json(args.normal_run/'provenance.json')
            for key in ('selection_sha256','model_name','revision','config_sha256','adapter_sha256'):
                if normal.get(key)!=provenance[key]: raise ValueError('Normal/fault comparison identity mismatch: '+key)
            if not normal['completed'] or normal['mode']!='run' or normal['subset']!='full':
                raise ValueError('Run fresh full normal evaluation before robustness')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'audit.json',audit)
    write_json(args.output/'leakage.json',leak)
    write_json(args.output/'provenance.json',provenance)
    started=time.monotonic()
    if args.mode=='archive':
        from src.evaluation.grpo_v1.archive import archived_pair
        chosen=subset(items,args.subset)
        before,after=archived_pair(chosen,run_dir,config)
        save_pair(args.output/'normal',before,after)
        provenance['source_sha256']={name:file_hash(run_dir/name) for name in ('base_dev/results.jsonl','eval_group_240/dev/results.jsonl')}
    elif args.mode in ('run','robustness'):
        from src.evaluation.grpo_v1.inference import load_evaluation_policy,fresh_pair
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        policy=load_evaluation_policy(args.snapshot,checkpoint,config)
        write_json(args.output/'runtime.json',policy.runtime_metadata)
        if args.mode=='run':
            save_pair(args.output/'normal',*fresh_pair(subset(items,args.subset),policy,config))
        else:
            for rate in (0.2,0.5):
                for seed in (2026,2027):
                    save_pair(args.output/f'timeout_{rate}_seed_{seed}',*fresh_pair(subset(items,'robustness'),policy,config,rate,seed))
        provenance.update(peak_vram_bytes=torch.cuda.max_memory_allocated(),independently_restored=True)
    provenance.update(completed=True,wall_seconds=time.monotonic()-started)
    write_json(args.output/'provenance.json',provenance)
    print(args.output)


if __name__=='__main__': main()
