# GRPO V2 Success-only Baseline preparation

Date: 2026-10-23 (Asia/Shanghai)

No formal 240-group training or Cost-aware Reward training was started. Existing V1 task data, configs, logs,
checkpoints, and result artifacts were not used as output targets.

## Local validation and GPU status

The bounded V2 smoke plan contains 24 tasks (8 each of L1/L2/L3), G=4, 96 training rollouts, one pass, and
checkpoints only at groups 0 and 24. It reuses the already-frozen V1 smoke selection but uses the aligned V2
sampling configuration: `do_sample=true`, `temperature=1.0`, `top_p=1.0`, and `top_k=0`. Hold-out evaluation is
disabled for smoke.

The local GPU preflight is blocked, not passed:

- `nvidia-smi` is unavailable.
- PyTorch reports `cuda_available=false`, no CUDA runtime, and no bf16 CUDA support.
- `bitsandbytes` and `peft` are not installed.
- The frozen Qwen2.5-7B-Instruct snapshot is not present at the required canonical path.

Consequently no GPU smoke result is reported. CPU dry-run reports `training_started=false`.

The complete CPU suite finished with **111 passed, 0 failed, 4 skipped** in 12.34 seconds. All four skips are
training-dependency checks blocked by the missing `peft` package (including the real adapter checkpoint reload
test). One urllib3/LibreSSL compatibility warning was emitted and did not fail a test.

## Cloud GPU smoke command

Run on a CUDA machine with at least approximately 24 GiB VRAM and the frozen model revision already available:

```bash
cd /path/to/ToolRL
.venv/bin/python -m pip install -r experiment/requirements-training.txt

MODEL_DIR=/path/to/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
RUN_DIR=experiment/results/grpo_v2/smoke_20261023_001

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  experiment/scripts/run_grpo_v2.py \
  --config experiment/configs/grpo_v2_gpu_smoke.json \
  --preflight --model-dir "$MODEL_DIR"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  experiment/scripts/run_grpo_v2.py \
  --config experiment/configs/grpo_v2_gpu_smoke.json \
  --train --model-dir "$MODEL_DIR" --output-root "$RUN_DIR"

.venv/bin/python experiment/scripts/run_grpo_v2.py \
  --config experiment/configs/grpo_v2_gpu_smoke.json \
  --verify-run --run-dir "$RUN_DIR"
```

The run fails closed unless all of the following hold: every stored old log-prob is finite and aligned to generated
tokens; pre-update ratios remain within 1e-4 of one; clip fraction is zero; approximate KL is at most 1e-7; every
mixed-reward group performs one optimizer step and has positive LoRA delta; every zero-variance group skips the
optimizer and has zero LoRA delta; the final checkpoint reload reproduces probe log-probs within 1e-4; required
output files exist; and every manifest size/SHA-256 matches the exact file inventory.

## Frozen hold-out test split

The report-only hold-out contains 72 tasks over six new world seeds, 3000–3005, with 24 tasks at each difficulty.
These seeds do not occur in train, dev, or diagnostic data. Each world uses a distinct `h*` entity namespace, and
the 12 templates introduce new lookup fields, conjunctive filters, grade/course combinations, candidate-limited
state changes, two-scholarship batches, verified no-op cases, and explicit pending-pair submissions. This is more
than a seed substitution of V1 templates.

Frozen assets include the generator source hash, manifest hash, per-world KB/task hashes, complete task list, exact
questions, template IDs, instance hashes, and expected-outcome hashes. Regeneration refuses to overwrite the data;
`--verify` performs byte-for-byte verification.

Overlap audit against every V1 train/dev/diagnostic world and task:

| Audit | Overlap | Denominator |
|---|---:|---:|
| Exact instance | 0 | 72 tasks |
| Exact question | 0 | 72 tasks |
| Normalized template | 0 | 72 tasks |
| Entity ID | 0 | 158 hold-out IDs |
| Expected answer/outcome | 12 | 72 tasks |

The 12 answer overlaps are scalar/category/count coincidences; instance, wording/template, and entity identity remain
distinct. The hold-out is explicitly prohibited for prompt changes, hyperparameter tuning, checkpoint selection, or
run selection in this phase.

## Recommended formal configuration and command

Only after the GPU smoke and its artifact verification pass, run the existing one-pass 240-group V2 configuration:

```bash
MODEL_DIR=/path/to/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
RUN_DIR=experiment/results/grpo_v2/success_only_240_001

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  experiment/scripts/run_grpo_v2.py \
  --config experiment/configs/grpo_v2_baseline.json \
  --preflight --model-dir "$MODEL_DIR"

# Deliberately not executed in this preparation round.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  experiment/scripts/run_grpo_v2.py \
  --config experiment/configs/grpo_v2_baseline.json \
  --train --model-dir "$MODEL_DIR" --output-root "$RUN_DIR"
```

The formal configuration remains Success-only (`task_success`), beta=0, one update per mixed group, no multi-epoch
replay, and no Cost-aware Reward.

## Remaining risks

- No QLoRA/NF4/bf16 path ran locally, so GPU memory, throughput, numerical stability, and adapter reload remain
  unverified here.
- Single-update ratios should start at one, making clipping mostly inactive by construction.
- Actual mixed-group frequency and optimizer-step count depend on real sampled rewards and cannot be inferred from
  dry-run validation.
- Hold-out answer overlap is nonzero for unavoidable common scalar/category outcomes; per-template performance will
  need confidence intervals before any generalization claim.
- The hold-out has not been used to compare checkpoints or experiments, by design.
