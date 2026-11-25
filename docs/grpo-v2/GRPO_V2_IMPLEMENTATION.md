# GRPO V2 correctness baseline

V2 is additive: it reuses the frozen V1 task selection and generated worlds but writes only to a new
`results/grpo_v2/<run>` directory. Existing V1 configs, logs, checkpoints, reports, and result directories are
never selected as output targets and are not rewritten.

## Correctness contract

Training sampling is fixed to `do_sample=true`, `temperature=1.0`, `top_p=1.0`, and `top_k=0`. Stored old
log-probabilities are computed in `raw_model_softmax`, so these neutral sampler transforms make the behavior
distribution and old-log-probability space agree. Config loading, policy construction, and collection each reject
an incompatible setting before an optimizer update.

Each mixed-reward group still receives exactly one optimizer update. No multi-epoch replay was added. Therefore
the current and old log-probabilities are normally identical when the loss is evaluated, importance ratios begin
near one, and clipping is expected to be mostly inactive. The metrics are still recorded because any departure is
a correctness signal, not because clipping is expected to drive this one-update experiment.

Per-group metrics include reward mean/variance, zero-variance status, token-weighted ratio mean/min/max, clip
fraction, non-negative approximate KL, ratio token count, and cumulative optimizer steps. The training summary
adds zero-variance group count and token-weighted aggregate ratio, clip fraction, and approximate KL.

## Artifacts

A completed V2 run contains the resolved config, selected task list, preflight and dry-run records, every training
rollout, per-group metrics, training summary, model/runtime metadata, adapter-only checkpoints plus optimizer/RNG
state, raw evaluation JSONL trajectories, status, report, and a SHA-256 artifact manifest. The manifest also records
all global/world/session seed derivation inputs. Output creation is exclusive and the CLI refuses paths outside the
V2 namespace.

CPU validation and dry-run checks do not establish that 7B NF4/bf16 GPU training reproduces any prior result.

## Cost-aware reward boundary

Cost-aware reward is intentionally not part of V2. The future insertion point is `task_reward(evaluation)` in the
collector: add a versioned reward adapter that consumes the existing evaluator fields (`task_success`, `tool_cost`,
and, if desired, rule violations), store both the unshaped task reward and every shaping component in each rollout,
then pass only the declared scalar into group normalization. Freeze the lambda/scale in a new V3 config and add
tests for ranking, zero-variance behavior, and reward-component manifesting before any training run.
