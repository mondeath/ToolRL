# Environment v1 模型规模诊断运行说明

## 仓库与冻结范围

真正的 Git 仓库根是 `experiment/`。Environment v1 已由 commit `258b989` 保存；原先位于 ToolRL 根目录的必要报告、baseline JSONL、任务、KB 和其他小型诊断记录，现已复制到此仓库的根目录和 `results/`。原文件保留。

模型权重/模型自带 tokenizer 随规模变化；按照最新要求，整组诊断现在统一默认使用 CUDA/bfloat16。dtype 可通过参数修改，但所有模型必须一致。固定任务文件 `configs/env_v1_diagnostic_tasks.json` 是有序 15 题，L1/L2/L3 各 5：L1 覆盖 GPA、列表、count；L2 覆盖资格与课程组合；L3 覆盖五个奖学金。所有模型使用相同 ID 和顺序，禁止按结果换题。

`configs/env_v1_model_scale.json` 固定三个模型 revision、KB/完整任务/prompt 哈希、原有行为源码哈希、生成和 rollout 配置。`run_model_scale.py` 在推理/报告前验证这些条件。新增统计只读取保存的轨迹，不修改 evaluator 或任务成功定义。

| 模型 | 固定 revision |
|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Qwen/Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |

3B/7B revision 来自 Hugging Face 官方仓库 metadata 查询；1.5B 沿用已有 baseline revision。模型下载与推理分开，不会把 1.5B 的 revision 误用于 3B/7B。

统一设置：failure_rate=0、do_sample=false、CUDA 默认 bfloat16、max_new_tokens=256、max_context_tokens=16384、max_steps=9、max_tool_calls=8、max_parse_retries=1、seed=2026。temperature=0.8、top_p=0.95 保持原配置值，但 greedy 模式不使用它们。没有新增可按模型调节的生成参数。

## 云端完整运行

在能以 bfloat16 运行 7B 推理的同一 CUDA 设备上顺序运行三种模型；不单独量化或改变某个模型精度。CUDA 不可用会报错，不会悄悄退回 CPU。

从 clone 后的仓库根目录执行（当前本机则先 `cd experiment`）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.lock.txt -r requirements-test.txt
pytest -q

# 首次联网下载固定 revision；缓存位于仓库上一级 .hf-cache/hub，不提交 Git。
python scripts/run_model_scale.py --download-only

# 与已归档 CPU 结果分开，三个模型都在 CUDA 上重新执行。
python scripts/run_model_scale.py --device cuda --dtype bf16 --output-root results/model_scale_cuda_bfloat16

# 仅重新生成对照报告，不调用模型。
python scripts/run_model_scale.py --report-only --dtype bf16 --output-root results/model_scale_cuda_bfloat16
```

`--dtype bf16` 与 `--dtype bfloat16` 等价；也可将整组显式切回 `--dtype float32`。CUDA 默认 bfloat16，CPU 默认 float32。原适配器固定 CPU=float32，因此显式 CPU+bf16 会在加载权重前报错，不会静默降级。summary 同时核对请求与实际 dtype；已有 float32 输出不能当作 bf16 跳过。

统一 runner 会锁定 15-task subset 和 failure_rate=0，逐模型调用同一 `run_env_v1.py`；每次完成 rollout 即 flush。失败时 `run.log` 保留 stderr 和原始命令。若存在部分结果，脚本拒绝覆盖；保留原目录并选新的 `--output-root` 重跑。已完成且配置匹配的模型可跳过，指标会从 JSONL 重新核对。不同设备的结果不会自动跳过或混为同配置实验。

每个模型目录包含：

```text
results.jsonl          # 原 trajectory + 离线附加统计
summary.json          # 汇总、配置、版本、选题和哈希
run.log               # 子进程命令、进度和错误
results.tasks.json    # 完整选中任务内容，仅供评测/复现
results.kb.json       # 固定 KB
results.prompt.json   # 第一题真实 messages 快照
```

`MODEL_SCALE_DIAGNOSTIC.md` 由保存的 JSONL/summary 生成。尚未运行的模型显示“—/待云端运行”；不把缺失数据当成 0%。最终只有所有模型的 device/dtype/torch/transformers 版本相同，报告才将它们视作同运行配置。

## 单模型或短诊断

在 ToolRL 工作区根目录可以直接执行用户要求的形式：

```bash
python experiment/scripts/run_env_v1.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda --dtype bf16 \
  --task-ids-file experiment/configs/env_v1_diagnostic_tasks.json \
  --output experiment/results/manual_7b/results.jsonl
```

也可用 `--task-ids v1-L1-00 v1-L2-00 v1-L3-00`，或 `--task-count 3` 限制前几题。显式 IDs 保持给定顺序，重复或不存在的 ID 会被拒绝。短诊断仅用于运行验证，不加入正式 15 题对照。直接 `run_env_v1.py` 的标准输出可由 shell 重定向到 `run.log`；推荐统一 runner 自动保存日志。

不传模型/设备参数时仍用原 CPU 1.5B 和原默认输出位置；显式模型默认输出到仓库内 `results/model_scale_<device>_<dtype>/<模型>/results.jsonl`，例如 `results/model_scale_cuda_bfloat16/qwen2.5-7b/results.jsonl`。统一 runner 的新推理也默认使用带设备/精度的目录；`--prepare` 仍使用历史归档目录 `results/model_scale`。直接 runner 仍保留原有覆盖输出行为，所以正式实验应使用会保护已有结果的 `run_model_scale.py`。

## 复用与统计口径

仓库内 `results/model_scale/qwen2.5-1.5b/` 是历史 CPU baseline 的固定子集，不是新推理；`provenance` 保存来源 JSONL SHA256。未把完整 40 题的运行耗时冒充 15 题耗时。3B/7B 空 JSONL、status=not_run 的 summary 和说明日志是明确的待运行占位。

```bash
# 不加载模型，仅检查并复用已有 CPU 子集、生成待运行目录和报告。
python scripts/run_model_scale.py --prepare
```

旧结果的 prompt、task、采样和工具预算与当前冻结版本相同，但设备/精度是 CPU/float32；本轮 CUDA/bfloat16 对照需要重新跑三个模型，包括 1.5B。若选择整组 CPU，原 1.5B 可复用，3B/7B 则需以同样 CPU/float32 和依赖版本运行。不要将不同设备或不同 dtype 的结果当成完全相同配置。原 float32 JSONL 和 summary 保留，未重标为 bf16。

- `success` 保持原 evaluator 的 task_success。
- `termination_reason` 直接来自 trajectory。
- `tool_calls` / `tool_cost` 保持原定义。
- `deep_search_calls` 是实际执行的 deep_search 数量。
- `repeated_identical_action_count` 是相邻已执行调用中 tool+arguments 完全相同的额外次数；字典键顺序无关。A,A,A=2；A,B,A=0。不是所有历史重复次数，也不统计未执行的提案。
- `answer_ready_but_continued` 为 null，表示未严格判定；采用上述重复调用统计作为用户允许的替代，不改 evaluator。

目前不能仅凭 1.5B 的结果判断原因。待 3B/7B 完成后，看成功率、终止情况、重复动作和成本是否共同改善，再决定是能力下限更突出还是协议/环境问题更突出。不修改环境难度、prompt、reward，不做 GRPO，不增加失败注入，不按单模型表现调参。
