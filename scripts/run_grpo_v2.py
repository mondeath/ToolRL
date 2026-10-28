"""Aligned GRPO V2 entry point. Formal GPU training requires explicit --train."""
from _common import ROOT
import argparse
import json
from pathlib import Path
import sys

from src.training.grpo_v1.baseline_runner import write_json
from src.training.grpo_v1.data import read_json
from src.training.grpo_v1.preflight import preflight
from src.training.grpo_v2.data import (dry_run, preflight_data, load_v2_data, validate_config,
    smoke_dry_run, smoke_preflight_data, load_v2_smoke_data, validate_smoke_config)
from src.training.grpo_v2.reporting import render_report, validate_artifact_manifest
from src.training.grpo_v2.runner import train_v2, train_v2_smoke, validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grpo_v2_baseline.json')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', choices=['dev', 'diagnostic'])
    mode.add_argument('--report', action='store_true')
    mode.add_argument('--verify-run', action='store_true')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    smoke = config.get('mode') == 'grpo_v2_gpu_smoke'
    validator = validate_smoke_config if smoke else validate_config
    loader = load_v2_smoke_data if smoke else load_v2_data
    preflight_loader = smoke_preflight_data if smoke else preflight_data
    dry_runner = smoke_dry_run if smoke else dry_run
    trainer = train_v2_smoke if smoke else train_v2
    validator(ROOT, config)
    if args.verify_run:
        if not args.run_dir or not args.run_dir.is_dir():
            parser.error('--verify-run requires an existing --run-dir')
        print(json.dumps(validate_artifact_manifest(args.run_dir), indent=2))
        return 0
    if args.output_root:
        intended = (ROOT / config['output_namespace']).resolve()
        resolved = args.output_root.resolve()
        try:
            resolved.relative_to(intended)
        except ValueError:
            parser.error('--output-root must be a new directory under ' + str(intended))
    if args.report:
        if not args.run_dir or not args.run_dir.is_dir():
            parser.error('--report requires an existing --run-dir')
        print(render_report(args.run_dir))
        return 0
    if args.output_root and args.output_root.exists():
        parser.error('--output-root must be new; V1/V2 runs are never overwritten')
    if args.dry_run:
        result = dry_runner(ROOT, config)
        if args.output_root:
            args.output_root.mkdir(parents=True)
            write_json(args.output_root / 'dry_run.json', result)
            write_json(args.output_root / 'status.json', dict(status='dry_run_passed', training_started=False))
            render_report(args.output_root)
        print(json.dumps(result, indent=2))
        return 0
    if args.eval:
        if not args.checkpoint:
            parser.error('--eval requires --checkpoint')
        validate_checkpoint(config, args.checkpoint, args.eval)
    if (args.train or args.eval) and not args.output_root:
        parser.error('A new --output-root under results/grpo_v2 is required')
    checks = preflight(ROOT, config, args.model_dir, data_loader=preflight_loader)
    if args.preflight:
        print(json.dumps(checks, indent=2))
        return 0 if checks['status'] == 'passed' else 2
    if checks['status'] != 'passed':
        print(json.dumps(checks, indent=2))
        return 2
    if args.train:
        print(json.dumps(trainer(ROOT, config, args.model_dir, args.output_root, checks), indent=2))
    else:
        from src.training.grpo_v1.policy import load_policy
        from src.training.grpo_v1.evaluation import greedy_eval
        _, dev, diagnostic = loader(ROOT, config)
        args.output_root.mkdir(parents=True)
        policy = load_policy(args.model_dir, config, checkpoint=args.checkpoint)
        result = greedy_eval(dev if args.eval == 'dev' else diagnostic, policy, config,
                             args.output_root / args.eval)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
