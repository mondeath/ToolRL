# Model-scale diagnostic：正式结果收尾

## 结论与下一步主模型

**后续 GRPO 主实验使用 Qwen2.5-7B-Instruct；1.5B / 3B 保留为 model-scale diagnostic baselines。当前 Environment v1 不继续降难度。**

同一固定 15-task subset 上，1.5B / 3B / 7B 分别成功 **1/15、2/15、11/15**。两个小模型的 L2/L3 均为 0/5；7B 达到 L1=5/5、L2=2/5、L3=4/5，并且没有 episode 因 `invalid_action` 终止。结果支持一个实用的能力门槛：在当前协议与预算下，1.5B / 3B 均未达到可靠执行 agent 任务的最低能力要求，7B 已跨过基础 tool-use protocol 门槛，剩余问题主要在任务约束、无效实体恢复、停止时机及调用成本。

这是对当前固定小样本的实验判断，不是“所有 3B 都不能做 agent”的普遍结论，也不能由三种规模定位精确参数量阈值。尤其 1.5B 使用历史 CPU/float32 和不同库版本；规模效应最有力的直接证据来自相同 CUDA/bfloat16、相同依赖版本的 **3B→7B：成功率提高 60 个百分点**。

本次仅读取已归档 JSONL、summary 和输入快照，重新汇总并分析；没有重新推理，没有执行工具轨迹，也没有重新评分。Environment v1、tasks、prompt、evaluator、reward 和 generation config 均未修改。

## 对比结果

每个模型 15 条，L1/L2/L3 各 5 条，ID 与顺序完全相同。成功/失败采用原 JSONL 的 `success`，不因答案部分正确、最终状态已达成或终止格式可理解而重新归类。

| Model | Overall | L1 | L2 | L3 | Invalid action rate | Max-tool-call rate | Repeated actions | Avg cost | Avg cost (success) | Avg cost (failure) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 6.7% | 20.0% | 0.0% | 0.0% | 20.0% | 73.3% | 78 | 27.800 | 6.000 | 29.357 |
| Qwen2.5-3B-Instruct | 13.3% | 40.0% | 0.0% | 0.0% | 66.7% | 20.0% | 6 | 4.933 | 1.000 | 5.538 |
| Qwen2.5-7B-Instruct | 73.3% | 100.0% | 40.0% | 80.0% | 0.0% | 13.3% | 13 | 7.600 | 6.182 | 11.500 |

- **Invalid action rate**：最终 `termination_reason=invalid_action` 的 episode 数 / 15，含既有一次格式修复耗尽后的失败；不是单次输出的 JSON 错误率。
- **Max-tool-call rate**：最终 `termination_reason=max_tool_calls` 的 episode 数 / 15。
- **Repeated actions**：沿用原定义，汇总相邻、完全相同的已执行 `tool_name + arguments` 的额外调用次数；JSON 键顺序不影响结果。A,A,A=2；A,B,A=0。不含解析失败或被预算阻止的提案，不把非相邻重复或参数略有变化的调用算进来。
- **Avg cost**：已执行工具的既定 cost 单位，不是推理时间、token 数或 GPU 费用；未执行的非法提案不收工具 cost。
- **Avg cost (success/failure)**：分别在原成功/失败 episode 集合中计算；空集合应为 N/A，而不是 0。本轮成功样本数依次为 1、2、11，失败样本数为 14、13、4。
- `answer_ready_but_continued` 保持未严格自动判定（null），未引入新 evaluator。下文的个别“已获答案/已完成状态后继续”是对保存轨迹的人工诊断，不替代成功率。

### 成本分解：3B 低平均成本不是高效率

| Model | 成功数 / cost 总和 | 失败数 / cost 总和 | 全部 cost 总和 |
|---|---:|---:|---:|
| 1.5B | 1 / 6 | 14 / 411 | 417 |
| 3B | 2 / 2 | 13 / 72 | 74 |
| 7B | 11 / 68 | 4 / 46 | 114 |

3B 的总体平均 cost=4.933 虽低于 7B 的 7.600，但 **10/15 episode 在 invalid_action 提前终止，其中全部 5 条 L2 都是零工具调用、零工具成本的失败**。其 10 条 invalid_action 轨迹总 cost=12，平均仅 1.200；另外 3 条预算耗尽轨迹各花费 20。这是失败造成的提前截断，不能解释为更高效地完成同样任务。

| Model | direct_answer episodes | invalid_action episodes | max_tool_calls episodes |
|---|---:|---:|---:|
| 1.5B | 1 | 3 | 11 |
| 3B | 2 | 10 | 3 |
| 7B | 13 | 0 | 2 |

3B 的成功均价=1 只来自两条基础 L1；7B 的成功均价=6.182 还包含两条 L2 和四条 L3，任务构成不同，不能据此断言 3B 成功时也更有效率。双方共同成功的 `v1-L1-00`、`v1-L1-02`，成本都为 1。7B 的失败均价为 11.500，高于其成功均价 6.182，说明错误恢复、冗余操作和不及时终止仍消耗明显成本。

重复次数同样受可执行轨迹长度影响：3B 的 6 次不能直接视为比 7B 的 13 次更好，因为多数 3B episode 尚未完成协议就退出。

## 数据来源与可比性

- 1.5B：[归档 diagnostic JSONL](results/model_scale/qwen2.5-1.5b/results.jsonl)，与 [40-task baseline](results/env_v1_baseline.jsonl) 按固定 ID 抽取的原记录逐字段核对一致。完整 40 题成功率是 2.5%，本表使用同题 15-task subset 的 6.7%，没有混用分母。
- 3B：[正式 CUDA/bf16 JSONL](results/model_scale_cuda_bf16/qwen2.5-3b/results.jsonl) 与 [summary](results/model_scale_cuda_bf16/qwen2.5-3b/summary.json)。
- 7B：[正式 CUDA/bf16 JSONL](results/model_scale_cuda_bf16/qwen2.5-7b/results.jsonl) 与 [summary](results/model_scale_cuda_bf16/qwen2.5-7b/summary.json)。
- 3B/7B 来自已拉取的正式结果提交 `00d5e65`；本报告没有改写输入结果。

| Model | Device / dtype | torch | transformers | Revision |
|---|---|---|---|---|
| 1.5B | cpu / float32 | 2.8.0 | 4.57.6 | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 3B | cuda / bfloat16 | 2.7.0a0+7c8ec84dab.nv25.03 | 4.57.1 | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| 7B | cuda / bfloat16 | 2.7.0a0+7c8ec84dab.nv25.03 | 4.57.1 | `a09a35458c702b33eeacc393d103063234e8bc28` |

三组归档中的 KB、任务和 prompt 哈希一致；3B/7B 的实际 KB/任务/messages 快照也与归档基准核对一致。任务文件为 [env_v1_diagnostic_tasks.json](configs/env_v1_diagnostic_tasks.json)。本次统计校验了原 summary、工具计费、重复次数及终止分布，均与 JSONL 重算一致。

统一配置为 failure_rate=0、do_sample=false、max_tool_calls=8、max_steps=9、max_parse_retries=1、max_new_tokens=256、max_context_tokens=16384、seed=2026；temperature=0.8/top_p=0.95 保持归档值，greedy 下不使用。1.5B 与云端两组存在设备、dtype、torch/transformers 版本差异，不能声称三组是纯参数量的严格对照；3B/7B 的记录则匹配上述运行条件。每级只有 5 题，单题即改变该级成功率 20 个百分点；只有一次 greedy rollout，不据此给出统计显著性或泛化保证。

## 1.5B：主要问题是无法可靠结束工具循环

1.5B 有 11/15 条预算耗尽、3/15 条 invalid_action，仅 1 条正常完成。L2/L3 全部预算耗尽；累计 80 次 deep_search、78 次连续重复动作，失败均价 29.357。

例如 `v1-L2-01` 首次 deep_search 已返回所需学生 ID，但继续执行相同查询，直到 8 次工具调用预算用完，未提交 direct_answer。L3 则持续检索资格，没有走到 submit。其失败更像“不能利用 observation 决定下一步/何时停止”，而不是数据库里不存在答案。

基础协议也仍不稳：`v1-L1-02` 在 lookup 后拼出混合字段的非法关系查询；`v1-L1-09` 构造 aggregate 时缺失正确参数。唯一成功的 `v1-L1-00` 还先把学生 s00 当成 scholarship 查询，触发一次 invalid_entity 后才用 lookup 获取 GPA，总 cost=6。1/15 的 task_success 不代表这条轨迹零违规。

## 3B：invalid_action 是主导失败，不能把少调用理解为进步

3B 的两条成功都是 L1。其 10 条 invalid_action 分布为 L1=3、L2=5、L3=2；全部五条 L2 在第一步的原始输出和修复尝试中都未能产出可执行动作，成本均为 0。

典型证据：

| Task | 保存轨迹中的错误 | 影响 |
|---|---|---|
| v1-L1-05 / v1-L1-08 | 用仅含 year 或 year/min_gpa 的 deep_search query | 缺少规定的完整关系查询条件，修复后仍 invalid_action |
| v1-L2-00 | 对 scholarships 表使用不存在的 year 字段 | 单表字段/schema 不匹配，尚未执行工具就失败 |
| v1-L2-01 / v1-L2-05 | 在 search.query 中生成 join | 超出既定单表 search schema；不是工具执行超时 |
| v1-L3-00 | 将 scholarship_id 和 year/min_gpa/course_id 两种 deep_search 格式合并 | Invalid relational query |
| v1-L1-09 / v1-L2-14 | 生成省略号、错误括号、转义或额外 filter 等结构 | JSON 或参数语义校验失败 |

总计 59 次模型输出中，20 次解析失败，分别对应上述 10 条 episode 的初次输出和一次修复；没有把现有解析器放宽来“挽救”结果。3B 的错误包含合法 JSON 但非法 schema，不能统称为 JSON 语法问题，也没有证据表明是模型文件缺失或缓存加载故障。

剩余三条失败 `v1-L3-03/06/12` 已能执行 search→deep_search→verify→submit，**保存的最终 applications 已齐全**；但仍反复提交，合计产生 9 次 duplicate_submit，最后预算耗尽而未 direct_answer，因此原 evaluator 判失败。这说明 3B 有部分多步执行能力，但不足以稳定遵守协议、避免重复写入并正确终止。6 次严格连续重复动作中，3 次是 deep_search、3 次是 submit；没有改变参数或交替调用的重复行为并不全部计入该指标。

## 7B：基础协议已可稳定执行，剩余错误有明确策略改进目标

7B 的 L1=5/5，且每条仅花费 1 个工具 cost。13/15 条以 direct_answer 正常结束，没有 invalid_action episode。75 次输出中，69 次直接通过严格解析，另 6 次由**原有**表面归一化恢复，归一化后没有解析失败。因此“基础 protocol 稳定”指当前既定 parser/normalizer 管线下的表现，并不意味着每次原始输出都是完全规范 JSON。

四条失败均有可定位的行为原因：

| Task | Cost | 原终止原因 | 失败说明 |
|---|---:|---|---|
| v1-L2-00 | 15 | direct_answer | 题目只要求列出资格名单，模型却 submit 了 s01/s02。最终答案 ID 正确，但只读任务修改了 applications，触发 illegal_state_mutation，必须保留失败判定。 |
| v1-L2-09 | 6 | direct_answer | deep_search 返回 s18/s19/s20；模型把已申请但仍符合资格的 s18 排除，回答只含 s19/s20，混淆“符合资格”和“尚未申请”。 |
| v1-L2-14 | 12 | max_tool_calls | 在已有候选结果后反复 verify 不存在的 scholarship `sch01`，未根据 invalid_entity 调整策略，耗尽预算。 |
| v1-L3-00 | 13 | max_tool_calls | submit 后保存的最终状态断言全为 true，却继续多次 verify，没有 direct_answer。不能因为状态已正确就把本轮失败改成成功。 |

另外两条 L2 虽成功，但 `v1-L2-01` 成本 16、`v1-L2-05` 成本 11，包含对无效 scholarship ID 的多次验证。7B 共 13 次连续重复，全部来自 verify。累计 16 项违规中，15 项为 invalid_entity，1 项为只读任务非法状态修改；11 条成功里有 2 条仍出现被拒绝的 invalid_entity，因此“成功且零违规”只有 9/15（60%）。原 evaluator 的 task_success 与零违规是不同指标，本报告保留该定义。

四条成功 L3 (`v1-L3-03/06/09/12`) 都能执行 deep_search→verify→submit→verify→direct_answer，成本均为 9，且无违规。7B 五条 L3 的保存最终状态断言均满足，其中一条仅因无法及时终止失败；这既证明状态任务可解，也保留了停止决策的优化目标。

## 为什么选 7B 做后续 GRPO 主模型

3B→7B 在相同云端运行条件下，成功从 2/15 升至 11/15，invalid_action 从 10/15 降至 0/15；差异不仅是工具调用数量。7B 已能产生大量真正完成的多步轨迹，同时仍有失败、高成本成功和违反任务约束的轨迹，能区分后续优化是否改善了实际策略。

小模型当前的零 L2/L3 成功与高协议/停止失败，会使训练结果更难解释为成本意识或任务策略的改进。选择 7B 能把主实验焦点放在：正确区分查询与写入、保留完整答案集合、处理无效实体、避免冗余验证、完成目标后及时结束。这里仅确定后续主模型，不实现 GRPO，不改 reward，也不把本轮 greedy 结果当成已经证明 GRPO 必然有效或组内采样一定有足够差异。

## 为什么 Environment v1 不需要继续降难度

7B 的 L1=100% 与 L3=80% 说明基础工具与状态流程已经可解，3B 的大量失败不能单独证明环境设计不可用。7B 的 L2=40%，L2/L3 合计为 6/10（60%）；加上 4/15 总体失败、重复 verify、无效实体和只读状态修改，成功、成本与约束之间仍有明显优化空间，远未饱和。

降低关系/状态要求，或为了把小模型分数抬高而放宽 parser、删除只读约束、延长预算、把“状态正确但未结束”算成功，会改变当前需要研究的策略问题。因此保留 Environment v1 和现有评分口径，以 7B 为后续主实验模型；1.5B / 3B 留作规模诊断基线。本轮不再为逼近某个预设成功率区间改任务。

## 离线复算与审计

在仓库根目录（当前工作区的 experiment/）执行：

```bash
python scripts/summarize_model_scale.py
python scripts/summarize_model_scale.py --json
```

该脚本只读取上述三个正式结果目录和已有归档/固定配置，打印主表或完整统计；不会下载/加载模型，不调用 environment.step，不调用 evaluator 重新判定成功，不覆盖任何结果，也不改写报告的分析文字。新增脚本与本报告是本次唯一提交内容。

输入 JSONL SHA256：

| Model | SHA256 |
|---|---|
| 1.5B | `16052da0ab182bc4d07587e29b1284e3dd0913d68f6a2025c344bde4131e7791` |
| 3B | `86d18b1d734b283db7e0ca22131ca380aa70accae23ea6bb49623958539bd9d1` |
| 7B | `8624485a0afa7dd1bd74ad74858823bfd7843aa92091c332d375549e734b4849` |
