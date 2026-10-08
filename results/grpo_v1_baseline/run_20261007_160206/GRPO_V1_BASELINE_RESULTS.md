# GRPO v1 task-only baseline results

状态：completed。

模型：Qwen2.5-7B-Instruct，NF4/bf16 + LoRA。reward仅为原evaluator task_success。
单次固定240题训练，G=4；dev为独立全部400题，diagnostic为固定15题，仅最终checkpoint评估。

选题hash：`77dd1d73f5cb17546988fe7c19be6a40788f01c5aa0d3cd47de939148fc9f1e9`。
计划：240 groups，960 sampled trajectories，最多240 optimizer steps。

## Dev：同一NF4 base vs 最终task-only GRPO

| 指标 | Base NF4 | GRPO task-only |
|---|---:|---:|
| overall success | 0.6725 | 0.9975 |
| L1 success | 0.8000 | 0.9900 |
| L2 success | 0.4133 | 1.0000 |
| L3 success | 0.8467 | 1.0000 |
| invalid_action rate | 0.0500 | 0.0000 |
| max-tool-call rate | 0.1825 | 0.0000 |
| repeated actions (total) | 234 | 2 |
| avg cost | 8.8925 | 5.7950 |
| avg cost successful | 6.9331 | 5.8045 |
| success & zero violation | 0.5250 | 0.9700 |
| rule violation count | 341 | 13 |
| deep_search usage | 0.7500 | 0.7500 |

## 最终diagnostic

| 指标 | 既有Base NF4（若提供） | 最终GRPO |
|---|---:|---:|
| overall success | — | 1.0000 |
| L1 success | — | 1.0000 |
| L2 success | — | 1.0000 |
| L3 success | — | 1.0000 |
| invalid_action rate | — | 0.0000 |
| max-tool-call rate | — | 0.0000 |
| repeated actions (total) | — | 0 |
| avg cost | — | 5.1333 |
| avg cost successful | — | 5.1333 |
| success & zero violation | — | 1.0000 |
| rule violation count | — | 0 |
| deep_search usage | — | 0.6667 |

没有提供可验证的既有NF4 diagnostic记录时，Base列保持缺失；不拿bf16基线替代，不在训练中运行diagnostic。

## 训练过程

- attempted groups：240
- mixed groups / rate：21 / 0.0875
- all-success groups / rate：215 / 0.8958
- all-failure groups / rate：4 / 0.0167
- optimizer steps：21
- reward distribution：{"1": 909, "0": 51}
- cumulative task success：0.9469
- success by difficulty：{"L1": {"trajectories": 320, "successes": 303.0, "success_rate": 0.946875}, "L2": {"trajectories": 320, "successes": 287.0, "success_rate": 0.896875}, "L3": {"trajectories": 320, "successes": 319.0, "success_rate": 0.996875}}
- peak VRAM bytes：8829834752
- wall-clock seconds（含加载/eval/checkpoint）：27546.1661

## Dev curve

| 已处理groups | optimizer steps | dev success | L1 | L2 | L3 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.6725 | 0.8000 | 0.4133 | 0.8467 |
| 60 | 15 | 0.9925 | 0.9900 | 0.9933 | 0.9933 |
| 120 | 16 | 0.9975 | 0.9900 | 1.0000 | 1.0000 |
| 180 | 20 | 0.9975 | 0.9900 | 1.0000 | 1.0000 |
| 240 | 21 | 0.9975 | 0.9900 | 1.0000 | 1.0000 |

本轮固定dev集success提高，变化+32.50个百分点。单次运行的描述性结果，不据此修改环境、任务或reward。
