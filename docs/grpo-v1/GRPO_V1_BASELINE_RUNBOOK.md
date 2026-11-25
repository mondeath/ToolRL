# Task-only GRPO baseline：云端手动运行

本轮代码和dry-run由Codex准备，GPU任务由你手动登录后执行。无需改SSH配置。沿用已经通过smoke的RTX3090、Python环境和本地Qwen2.5-7B snapshot；不调超参、不重新生成数据、不从smoke adapter继续训练。正式baseline从同一个NF4 base的全新LoRA开始。

本机验证：**103 tests passed**（含真实tiny LoRA连续5次更新、240-group完整流程的CPU合成测试）。没有启动正式GPU训练，没有更改toolrl-cloud配置。

## 固定实验与资源计划

| 项目 | 固定值 |
|---|---|
| 训练集 | 240题，L1/L2/L3各80 |
| world分布 | train seeds1000–1029各8题；每个25%区间每个world恰好2题 |
| 顺序 | configs/grpo_v1_baseline_tasks.json，预先固定，无模型表现参与 |
| selection hash | 77dd1d73f5cb17546988fe7c19be6a40788f01c5aa0d3cd47de939148fc9f1e9 |
| pass / group size | 1 / 4 |
| 训练groups / trajectories | 240 / 960 |
| optimizer steps | 每个mixed group一次，最多240；全0/全1组不step也不重采样 |
| checkpoint位置 | 已处理groups 0、60、120、180、240；不以optimizer steps作百分比 |
| dev | 全部400题，seeds2000–2009，100 L1 / 150 L2 / 150 L3 |
| dev eval时点 | base + 25% + 50% + 75% + final，共2000条greedy trajectory |
| diagnostic | 原固定15题，仅最终checkpoint一次 |
| eval总数 / 总trajectory数 | 2015 / 2975 |

按用户提供的smoke mixed rate≈0.417外推，约100次update，但这不是停止条件，实际每个mixed group都更新。

暂估总时长约 **5–20小时**。这是未经实际3090分阶段耗时校准的资源规划：假设每训练group（含其update开销）45–180秒，每greedy episode3–12秒，计算结果为4.68–18.72小时；另有加载、checkpoint磁盘IO。序列长度和GPU占用会改变耗时，不能把此范围当作实测保证。完整dev评估占用较多时间，本轮不为缩短时间换小dev或调整协议。

NF4 4-bit + double quant、bf16 compute、LoRA r16/alpha32/dropout0、AdamW lr1e-5/weight_decay0、max_grad_norm1、gradient checkpointing不变。训练sample=true、temperature0.8、top_p0.95、top_k0；eval greedy。reward仅原evaluator task_success；KL关闭。OOM停止并保存错误状态，不自动降参数或重启。

## 1. 进入已完成smoke的仓库与环境

手动登录云端，在原smoke使用的Python环境中，进入**包含scripts/和configs/的ToolRL Git根目录**。下面命令都从此目录运行。仓库路径按你云端实际位置，不假设为本机的experiment/目录。

建议先进入tmux会话，再依次执行下面全部命令，避免退出SSH中断长跑。已有smoke的本地改动应保留；不要用reset --hard覆盖。

```bash
git status --short
git pull --ff-only origin main
git log -1 --oneline
python --version
python -m pytest -q tests/test_grpo_v1_baseline.py tests/test_grpo_v1_core.py tests/test_grpo_v1_training.py
```

这些测试不训练7B。使用已经验证过的smoke依赖环境；不要在这轮顺便升级torch/Transformers/PEFT/bitsandbytes。

## 2. 固定数据检查 / dry-run

```bash
BASELINE_STAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_RUN="results/grpo_v1_baseline/run_${BASELINE_STAMP}"
MODEL_DIR=/root/shared-nvme/hf-cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

python scripts/run_grpo_v1_baseline.py --dry-run \
  --output-root "results/grpo_v1_baseline/dry_run_${BASELINE_STAMP}"
```

MODEL_DIR应指向原smoke实际使用的同revision标准HF缓存路径；若你使用另一缓存根，只调整路径，不换权重。dry-run必须输出passed、240 groups、960 training rollouts、240 maximum_optimizer_steps、train_eval_overlap=false。选题hash应与上表一致。

## 3. GPU preflight（不加载7B、不训练）

```bash
python scripts/run_grpo_v1_baseline.py --preflight \
  --model-dir "$MODEL_DIR" \
  --output-root "results/grpo_v1_baseline/preflight_${BASELINE_STAMP}"
```

必须status=passed再进入下一步；检查GPU型号/显存、bf16、依赖版本、snapshot可读、冻结hash和split隔离。preflight不验证NF4 kernel实际反传，已完成的3090 smoke是这项依据。

## 4. 一次正式运行

程序启动后再次执行同一dry-run，保存step0 adapter，跑NF4 base dev400；之后遍历全部240 groups并在60/120/180/240处自动保存checkpoint和dev评估，最后自动跑固定15题diagnostic。到3次optimizer step不会停止更新。

```bash
set -o pipefail
python -u scripts/run_grpo_v1_baseline.py --train \
  --model-dir "$MODEL_DIR" \
  --output-root "$BASELINE_RUN" \
  2>&1 | tee "${BASELINE_RUN}.console.log"
```

如需在最终diagnostic表中同时展示**已有smoke的NF4 base15题**，可在上述train命令追加：

```bash
--base-smoke-dir /实际路径/已完成的_smoke_run
```

该目录必须包含`config.json`和`before_nf4/results.jsonl`、`before_nf4/summary.json`；代码核对模型、量化、采样、协议和15题顺序后复用记录。没有该文件时，仅该对照列为空；dev仍有本轮实测NF4 base vs final的完整比较。不使用bf16结果冒充NF4，不额外提前运行diagnostic。

运行日志：

```bash
tail -n 3 "$BASELINE_RUN/metrics.jsonl"
cat "$BASELINE_RUN/training_summary.json"
cat "$BASELINE_RUN/status.json"
```

每组保存rewards/advantages、difficulty、三种group类型、是否step、loss/grad/lr、trajectory长度、invalid/max_tool_calls、重复动作、cost/violations、生成token数和GPU内存。汇总持续更新累计success、分难度success、三种group计数/率、optimizer steps及reward分布。所有诊断字段均不进入reward。

## 5. 中途checkpoint与dev eval

主进程自动完成，无需另开并发GPU任务：

| 进度 | adapter + optimizer/RNG | dev结果 |
|---|---|---|
| 0% | checkpoints/group_000/ | base_dev/ |
| 25% | checkpoints/group_060/ | eval_group_060/dev/ |
| 50% | checkpoints/group_120/ | eval_group_120/dev/ |
| 75% | checkpoints/group_180/ | eval_group_180/dev/ |
| final | checkpoints/group_240/ | eval_group_240/dev/ |

这些目录在`$BASELINE_RUN`下。只保存PEFT adapter、tokenizer和必要状态，不保存完整base；每次训练中checkpoint保存后原位重载并检查logprob一致。

如训练已停止且需要**单独复核**某个中途checkpoint，在同一3090空闲时执行（正常运行不必重复）：

```bash
python scripts/run_grpo_v1_baseline.py --eval dev \
  --model-dir "$MODEL_DIR" \
  --checkpoint "$BASELINE_RUN/checkpoints/group_060" \
  --output-root "${BASELINE_RUN}_dev060_recheck"
```

50%/75%对应替换为`group_120`/`group_180`并使用新的output-root。这些命令只eval、不更新参数，不用于挑选checkpoint或超参；最终比较始终使用group240。

## 6. 最终diagnostic eval

正常train已经自动执行一次，结果在`$BASELINE_RUN/eval_group_240/diagnostic/`；**已有结果无需重跑**。

若最终checkpoint已保存，但最终eval因外部中断没有完成，可用独立进程手动执行：

```bash
python scripts/run_grpo_v1_baseline.py --eval diagnostic \
  --model-dir "$MODEL_DIR" \
  --checkpoint "$BASELINE_RUN/checkpoints/group_240" \
  --output-root "${BASELINE_RUN}_final_diagnostic_recheck"
```

脚本在加载模型前检查checkpoint的attempted_groups必须是240；对group060/120/180请求diagnostic会直接拒绝。独立eval输出位于新的目录，不覆盖原训练产物，也不会自动把中断运行标记为完整成功。

## 7. 报告与验收

```bash
python scripts/run_grpo_v1_baseline.py --report --run-dir "$BASELINE_RUN"
cat "$BASELINE_RUN/GRPO_V1_BASELINE_RESULTS.md"
```

正常运行会自动生成/更新同一报告。报告比较dev上Base NF4 vs最终GRPO：overall/L1/L2/L3 success、invalid_action、max-tool-call、重复动作、avg cost、avg cost successful、success & zero violation、rule violation count、deep_search usage；另列最终15题、训练mixed rate/optimizer steps/reward分布、dev curve、peak VRAM和含eval/IO的wall-clock。

报告只记录success提高/不变/下降，不据结果换任务、调整reward或改变环境。status=completed、attempted_groups=240、optimizer_steps=mixed_group_count、base_unchanged=true，以及所有5个checkpoint和最终eval产物构成本次完成证据；如果没有有效adapter更新会明确标记，不伪造改善。

失败时查看status.json和console.log。程序不自动降低设置、不重采样、不多epoch，不提供隐式resume；已完成checkpoint保留可评估，重启或续训需另行决定，不能把部分运行当作本次完整baseline。

本机dry-run报告：`results/grpo_v1_baseline/dry_run_001/dry_run.json`。正式GPU结果尚未生成，本仓库顶层GRPO_V1_BASELINE_RESULTS.md当前仅记录准备状态；实际云端报告生成在上面的run目录。
