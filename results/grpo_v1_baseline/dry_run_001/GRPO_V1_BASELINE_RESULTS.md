# GRPO v1 task-only baseline results

状态：dry_run_passed。

模型：Qwen2.5-7B-Instruct，NF4/bf16 + LoRA。reward仅为原evaluator task_success。
单次固定240题训练，G=4；dev为独立全部400题，diagnostic为固定15题，仅最终checkpoint评估。

选题hash：`77dd1d73f5cb17546988fe7c19be6a40788f01c5aa0d3cd47de939148fc9f1e9`。
计划：240 groups，960 sampled trajectories，最多240 optimizer steps。

## Dev：同一NF4 base vs 最终task-only GRPO

| 指标 | Base NF4 | GRPO task-only |
|---|---:|---:|
| overall success | — | — |
| L1 success | — | — |
| L2 success | — | — |
| L3 success | — | — |
| invalid_action rate | — | — |
| max-tool-call rate | — | — |
| repeated actions (total) | — | — |
| avg cost | — | — |
| avg cost successful | — | — |
| success & zero violation | — | — |
| rule violation count | — | — |
| deep_search usage | — | — |

## 最终diagnostic

| 指标 | 既有Base NF4（若提供） | 最终GRPO |
|---|---:|---:|
| overall success | — | — |
| L1 success | — | — |
| L2 success | — | — |
| L3 success | — | — |
| invalid_action rate | — | — |
| max-tool-call rate | — | — |
| repeated actions (total) | — | — |
| avg cost | — | — |
| avg cost successful | — | — |
| success & zero violation | — | — |
| rule violation count | — | — |
| deep_search usage | — | — |

没有提供可验证的既有NF4 diagnostic记录时，Base列保持缺失；不拿bf16基线替代，不在训练中运行diagnostic。

## 训练过程

- attempted groups：0
- mixed groups / rate：0 / —
- all-success groups / rate：0 / —
- all-failure groups / rate：0 / —
- optimizer steps：0
- reward distribution：{}
- cumulative task success：—
- success by difficulty：{"L1": {"trajectories": 0, "successes": 0, "success_rate": null}, "L2": {"trajectories": 0, "successes": 0, "success_rate": null}, "L3": {"trajectories": 0, "successes": 0, "success_rate": null}}
- peak VRAM bytes：—
- wall-clock seconds（含加载/eval/checkpoint）：—

## Dev curve

| 已处理groups | optimizer steps | dev success | L1 | L2 | L3 |
|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — |
| 60 | — | — | — | — | — |
| 120 | — | — | — | — | — |
| 180 | — | — | — | — | — |
| 240 | — | — | — | — | — |

正式运行尚未完整结束，不能回答task-only GRPO是否改善policy；缺失结果不是0分。
