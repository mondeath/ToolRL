"""Fixed 240-task, one-pass task-only GRPO baseline; explicit modes only."""
from _common import ROOT
import argparse
import json
from pathlib import Path
import sys

from src.training.grpo_v1.data import read_json
from src.training.grpo_v1.baseline_data import load_baseline_data, dry_run, preflight_data
from src.training.grpo_v1.baseline_runner import write_json, train_baseline, validate_checkpoint
from src.training.grpo_v1.baseline_reporting import render_report
from src.training.grpo_v1.preflight import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/grpo_v1_baseline.json')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', choices=['dev', 'diagnostic'])
    mode.add_argument('--report', action='store_true')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-dir', type=Path, help='Existing run for --report')
    parser.add_argument('--checkpoint', type=Path, help='Baseline adapter checkpoint for --eval')
    parser.add_argument('--base-smoke-dir', type=Path, help='Optional existing smoke run containing before_nf4 diagnostic artifacts')
    args = parser.parse_args()
    config = read_json(args.config)
    if args.report:
        if not args.run_dir or not args.run_dir.is_dir():
            parser.error('--report requires an existing --run-dir')
        print(render_report(args.run_dir))
        return 0
    if args.checkpoint and not args.eval:
        parser.error('--checkpoint is only valid for standalone eval, not implicit training resume')
    if args.output_root and args.output_root.exists():
        parser.error('--output-root must be new; previous runs are never overwritten')
    if args.dry_run:
        result = dry_run(ROOT, config)
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
        validate_checkpoint(ROOT, config, args.checkpoint, args.eval)  # Before loading GPU/model.
    if (args.train or args.eval) and not args.output_root:
        parser.error('A new --output-root is required')
    result = preflight(ROOT, config, args.model_dir, data_loader=preflight_data)
    if args.preflight:
        if args.output_root:
            args.output_root.mkdir(parents=True)
            write_json(args.output_root / 'preflight.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'passed' else 2
    if result['status'] != 'passed':
        print(json.dumps(result, indent=2))
        return 2
    if args.train:
        plan = dry_run(ROOT, config)
        print('Pre-training dry-run: ' + json.dumps(plan), flush=True)
        print(json.dumps(train_baseline(ROOT, config, args.model_dir, args.output_root, result, args.base_smoke_dir), indent=2))
    else:
        from src.training.grpo_v1.policy import load_policy
        from src.training.grpo_v1.evaluation import greedy_eval
        _, dev, diagnostic = load_baseline_data(ROOT, config)
        args.output_root.mkdir(parents=True)
        policy = load_policy(args.model_dir, config, checkpoint=args.checkpoint)
        summary = greedy_eval(dev if args.eval == 'dev' else diagnostic, policy, config, args.output_root / args.eval)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
