# ToolRL：工具调用 Agent 能否答对，而且少花成本？

同一道题，Agent 可以直接查到答案，也可能重复检索后才答对。用 GRPO 训练 Qwen2.5-7B-Instruct 时，**在成功奖励中扣除工具成本，能否让模型在新题上用更便宜的路径答对？**

| 模型 | 固定开发集：成功 / 400 | 新模板盲测：成功 / 72 |
|---|---:|---:|
| Base（未训练） | 246（61.50%） | 22（30.56%） |
| Success-only（只奖励成功） | 399（99.75%） | 35（48.61%） |
| Cost-aware（成功时再扣成本） | 400（100%） | 35（48.61%） |

**开发集上的节省没有在盲测复现。** 两种训练模型在盲测中答对的是**同样的 35 题**。在共同答对题上，Cost-aware 相比 Success-only 在开发集平均少花 **0.343** 个工具成本单位，在盲测却平均**多花 0.457**（119 对 103，合计多花 16）。盲测全题均值略低，是因为它在两者都失败的 37 题上花得更少，不是因为它更省地完成了任务。

**如何对照：** 项目把早期只读查询环境扩展为含状态修改、违规检查和最终状态判分的工具环境；这是独立的环境工程。上述两种训练均使用同一冻结环境、底座、题目、提示、工具价格与预算，**实验变量是训练奖励，不是判分器**。每种方法用 240 题 × 4 条采样，固定取第 240 组 checkpoint；仅一个训练种子，72 道盲测题共享六个测试世界。

[查看配对报告：分母、成本总数与复算方法](reports/toolrl-paired-results.md) · [环境设计与任务示例](ENV_V1_PLAN.md)

以下是早期原型的运行方法和实现记录，不是上述正式对照的复现命令。

## 运行

从仓库根目录运行：

```bash
python3 scripts/run_smoke_test.py
python3 -m unittest discover -s tests -v
```

也可分步生成数据：

```bash
python3 scripts/generate_kb.py
python3 scripts/generate_tasks.py
```

配置位于 `configs/default.json`：seed=42、search top_k=5、每轮最多 12 次工具调用、cost_lambda=0.1。输出路径始终相对于 experiment，不依赖当前工作目录。再次运行会覆盖生成文件。

## 目录和职责

```text
experiment/
├── README.md
├── configs/default.json
├── data/
│   ├── kb.json
│   ├── tasks.public.json
│   ├── tasks.evaluation.json
│   ├── trajectories.jsonl
│   ├── example_trajectory.json
│   └── evaluation_summary.json
├── src/
│   ├── kb/             # 固定种子生成世界，引用及双向关系检查
│   ├── tasks/          # 64 条题目；公开视图与评估字段分离
│   ├── tools/          # 单字段查询、限量检索、关系路径查询
│   ├── environment/    # episode 生命周期、工具调度、错误和成本日志
│   ├── trajectories/   # 统一 dataclass schema 与 JSONL 导出
│   ├── rewards/        # AIReward Protocol、Mock、Pure 和 CostAware
│   └── evaluation/     # 正确率、成本、调用率与逐轨迹奖励明细
├── scripts/
│   ├── _common.py
│   ├── generate_kb.py
│   ├── generate_tasks.py
│   └── run_smoke_test.py
└── tests/test_pipeline.py
```

世界包含 30 students、8 courses、5 professors、5 projects、4 clubs，另有 8 schedules 和 4 classrooms。虚拟 ID/名称按序产生，属性与关系用局部随机数生成器生成。课表使用不冲突的教室/时间槽。验证 advisor 等外键，以及 enrollment、project、club 和 schedule 双向关系。

任务组成：16 single-hop、12 list/filter、16 two-hop、20 multi-condition。列表题答案是实体 ID 集合，空集合也是合法答案。`tasks.evaluation.json` 保存 ground_truth、query_plan 和 underlying_facts；`tasks.public.json` 仅包含 task_id/question。环境 reset 拒绝包含额外评估字段的任务对象。

## Action 协议

```python
env.reset(public_task(task))
env.step('lookup', entity='s00', field='advisor')  # cost=1
env.step('search', query='Student-')              # cost=2，keyword 检索
query = {
    'table': 'students',
    'where': [{'field': 'id', 'value': 's00'}],
    'select': 'advisor.department',
    'reduce': 'scalar',
}
env.step('deep_search', query=query)              # cost=8，实际遍历 KB
env.step('direct_answer', final_answer='Logic')    # cost=0，结束 episode
```

`search` 也支持结构化 query，限单表字段；`deep_search` 支持关系路径，例如 `courses.professor.department`。where 条件按 AND 组合，每个条件对关系展开值采用 any 语义。运算符为 eq（默认，标量相等/列表成员）、gte、contains（忽略大小写的子串）。默认 select=id、reduce=list；scalar 要求恰好一个去重值。多条件跨列表路径不提供同一关联对象绑定语义。

search 最多返回 top_k 个实体，返回 truncated 和 matched_count；结果不完整时不能将局部列表当成全集。deep_search 返回完整结果及查询实际读取的字段证据，包括用于排除候选项的字段。两种工具均直接读取 KB，不读取 tasks。查询支持确定性的结构化 DSL，不解析任意自然语言。

direct_answer 既可以作为第一步，也可以在工具之后提交答案；它记录在 steps 中，不计入 tool_calls。已知工具参数错误也记录 observation/error 并收取原工具成本；未知 action 被拒绝。达到调用上限后仍可提交答案。结束后继续调用需要 reset。环境复制 KB 和返回值，避免 observation 被外部修改后改变日志。

## Reward 与评估

AIReward 接口为 `score(question, final_answer, evidence) -> float`。当前 evidence 使用原始 observations，不传入 cost、tool_name 或 ground_truth。MockAIReward 只检查答案是否匹配某个 observation 的 answer/value，用于验证管线，不能充当任务质量裁判：部分答案可能得到满分，无证据的正确回答也可能得零分。

PureReward 返回 AI 分数；CostAwareReward 返回 AI 分数减去 lambda × tool_call 成本之和。breakdown 同时输出 ai_reward、total_cost、cost_penalty、pure_reward、cost_aware_reward；缓存的 trajectory.total_cost 必须与调用求和一致。

exact_accuracy 使用 JSON 值的严格比较，列表保留顺序；normalized_accuracy 对字符串做 NFKC、忽略大小写、折叠空白，列表按无序去重集合比较，不解析字符串里的 JSON/自然语言。deep_search_usage_rate 的分母是轨迹数量（至少调用一次的比例），另输出以所有工具调用为分母的 deep_search_call_fraction。direct_answer 不计入 average_tool_calls。评估需要已完成轨迹；空输入报错。

## Smoke test 的含义

每条任务构造三条轨迹：economical 用 lookup 或 search，检索截断时回退 deep_search；redundant_deep 重复两次 deep_search；direct_abstain 不调用工具并回答不知道。最终答案来自工具 observation，未拷贝 ground_truth。这个人工测试驱动器读取参考 query_plan，明确标记 scripted_reference_plan=true；它不是可用于实际评测的 policy。

默认生成 192 条轨迹。economical 和 redundant_deep 正确率均为 100%，Pure reward 均为 1；平均成本分别为 3.125 和 16，CostAware reward 分别为 0.6875 和 -0.6。整体准确率为 2/3，因为包含 64 条直接弃答轨迹。该结果只验证成本惩罚和完整链路，不证明模型会过度使用工具。

完整示例见 `data/example_trajectory.json`；所有轨迹见 `data/trajectories.jsonl`；汇总及逐条 reward breakdown 见 `data/evaluation_summary.json`。

## 原型阶段的设计记录（历史）

- policy 只接收 public task 和 observation，不能访问 evaluation 文件或 smoke 驱动器的 query_plan；当前隔离是 Python 接口约束，并非进程安全沙箱。
- 将模型输出解析成明确 action schema；补充 token/action mask、logprob、终止原因与截断状态。当前 Trajectory 只存结构化行为，不保存训练张量。
- 同一题的多条 rollout 使用相同 KB/config 和各自独立环境；明确失败调用、超时、达到预算但未提交答案的奖励规则。当前达到预算后要求显式 direct_answer。
- reward 是 episode 末尾标量，成本逐调用记录；实现 GRPO 前明确 group normalization 与 lambda 的尺度，避免标准化抵消成本信号。默认 lambda 只是示例，当前甚至使弃答优于重复昂贵查询。
- Mock 必须替换为真正关注完整性和正确性的质量评分器。建议统一 evidence 格式、去重并限制长度，控制更多 evidence 对裁判评分的影响；评分器不要接收成本。保留独立 ground-truth accuracy 检查 reward hacking。
- 固定训练/验证 task split、KB seed、judge 版本和配置快照；64 条模板题仅用于功能验证，不支持泛化或因果结论。


## 真实 HuggingFace policy 与多轨迹采样

新增模块：

```text
configs/policy.json                  # 正式配置：12 题 × G=4（48 条）
configs/policy_cpu_smoke.json        # CPU 极小验证：4 题 × G=2（8 条）
requirements-inference.txt           # 可选 torch / transformers 依赖
src/policies/
  __init__.py                       # Policy / PolicyOutput / AgentState
  schema.py                         # 公开 action/数据库字段 schema、system prompt
  parser.py                         # JSON、action 名称和参数验证
  huggingface.py                    # AutoTokenizer + AutoModelForCausalLM
src/rollouts/__init__.py             # rollout、rollout_group、独立 session/seed
src/evaluation/policy.py             # 离线 accuracy、使用分布、组内多样性、报告
scripts/run_policy_rollouts.py       # 真模型执行入口；没有 mock fallback
reports/                            # 自动生成的人类可读报告（含 5 条完整轨迹）
tests/test_policy_rollouts.py        # parser、修复、截断、隔离、统计回归测试
```

安装与执行（仓库根目录）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
# 首次运行会下载配置的模型；正式采样面向 CUDA。
.venv/bin/python scripts/run_policy_rollouts.py
# 本地已有缓存后，CPU smoke 不需要联网：
.venv/bin/python scripts/run_policy_rollouts.py --config configs/policy_cpu_smoke.json
# 保持测试无需 torch/transformers：
python3 -m unittest discover -s tests -v
```

正式输出为 `data/policy_rollouts.jsonl`、`data/policy_rollout_summary.json` 和 `reports/base_policy_analysis.md`；`data/policy_prompt_example.json` 保存实际首轮 messages。
CPU 输出独立放在 `data/cpu_smoke/` 和 `reports/base_policy_analysis_cpu_smoke.md`，避免冒充正式实验。
每完成一条 rollout 即写入并 flush JSONL；中断进程可保留此前完成的轨迹，自动续跑和并行推理暂未实现。再次运行相同输出路径会覆盖文件。

模型默认 Qwen2.5-1.5B-Instruct，使用固定 commit revision；可改为本地路径或其他 1.5B–3B instruct 模型（同时调整 revision）。支持 `device=auto/cuda/cuda:0/cpu`；auto 优先 CUDA，否则 CPU，显式 CUDA 不可用时由 cpu_fallback 决定是否回退。CPU 强制 float32，并默认拒绝超过 8 条的 CPU 采样（使用 CPU smoke 配置）；实际 device/dtype/依赖版本和 resolved model revision 写入 metadata。配置控制 temperature、top_p、max_new_tokens、max_context_tokens、do_sample 和 cpu_threads；无量化、LoRA 或训练逻辑。

底层使用 tokenizer 自带 `apply_chat_template` 构造 prompt，通过 `model.generate` 在 `torch.inference_mode()` 下采样，仅解码新生成 token。参考 [HF chat templates](https://huggingface.co/docs/transformers/chat_templating) 和 [Qwen 模型接口说明](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。

### Prompt 和隔离边界

完整 system prompt 在 `src/policies/schema.py`，用户消息结构为：

```text
Question:
<public question>

Available tools:
<四个 action 的参数格式 + 数据库字段/关系 schema，不包含记录>

History:
<此前 action 与 observation；若修复失败格式，包含原始输出和格式反馈>

Choose the next action.
```

system prompt 说明 lookup 的单实体/字段能力、search 的 top-k/单表限制、deep_search 的结构化多跳查询，以及 observation 在下一轮返回、允许多轮工具调用、必须用 direct_answer 结束。**没有提供工具成本、reward、ground_truth 或参考 query_plan**。

runner 只接受恰好包含 task_id/question 的公开对象。AgentState 只有 question/history/action_schema，history 是 allowlist 重建，不直接序列化 trajectory（后者含成本）。task_id 仅用于日志和 seed，不传给模型。脚本为均衡抽取各 task_type，在 orchestrator 端读取类型标签；参考答案只在 rollout 结束后的离线评估中使用，Policy/runner 不接收 evaluation 对象。

这属于应用接口隔离；任意恶意 Python policy 仍有同进程文件访问能力。后续需要更强隔离时可将推理与评估拆成不同进程/权限目录。

### Parser、限制与终止

只接受一个完整 JSON 对象；可移除包住整个输出的 JSON code fence，不从说明文字中猜取 JSON，不修补字段或答案。拒绝重复 key、非有限数值、未知 action、额外参数、非法路径和类型。最终答案为 JSON 标量或标量数组。

每个决策最多一次格式修复（max_parse_retries=0 或 1）；修复仅返回 parser 错误与 schema，不提供答案提示。每次尝试分别记录 raw_outputs 和 parse_status，包含 success、parsed_action、invalid_arguments、error、retry_count。

- direct_answer：正常结束；即便回答错误也正常结束，由离线评估判断正确性。
- invalid_action：修复耗尽仍解析失败；保存失败记录，无伪造答案，不标记预算截断。
- max_steps：决策轮数耗尽；truncated=true。修复次数不单独占决策轮数。
- max_tool_calls：预算用完后再尝试工具时截断，不执行、不计费。预算用完仍可提交 direct_answer。
- context_limit / policy_error：上下文或推理失败，保留已有轨迹并标记 truncated，不静默删历史。

解析通过但查询失败（例如实体不存在）按旧环境执行：返回 error observation，收取原工具成本，模型可在下一轮自行处理。它与 parser invalid_arguments 分开统计。

新增 trajectory 字段均有默认值，不改变旧调用方式。tool_sequence 记录已执行工具以及 direct_answer；不包括无效/被预算阻止的 action。number_of_steps 是决策轮数，raw_outputs 长度是实际生成次数（含 repair）。token_ids、action_token_mask、logprobs 暂留空；不能直接用于训练。

### 组内随机性与统计口径

`rollout_group(task, policy, env_factory, config, G=4)` 每次创建独立 env。每条 rollout 通过 base seed、task_id、rollout_id 派生 seed；同一 session 再按生成次数派生子 seed，格式修复也独立采样。共享的只有只读模型权重/tokenizer，每次 generate 从完整 history 重新构建输入，不跨 rollout 复用 KV cache。

当前顺序运行，在 torch RNG fork 上下文内保存/恢复随机状态，不支持多线程并发共享同一个模型。相同配置/版本/设备可重复；不保证跨 CUDA/CPU 的位级一致性。不同 seed 不保证产生不同 action。

准确率将所有 invalid/truncated 轨迹计入分母并判为失败；保持原 normalized answer 规则。报告同时给出 task level、按 task_type 汇总、first_action_counts、action_usage（含 answer）、tool_call_usage（只含工具）、invalid action attempts / all attempts、episode invalid rate 和 truncation rate。

组内 unique_tool_sequences 使用已执行 action 序列（含 direct_answer）；unique_final_answers 使用归一化后答案，仅统计真实提交的答案，无答案另外记录。cost_variance 使用总体方差。报告仅诊断 ground-truth accuracy，**不调用 MockAIReward、AI Judge 或 CostAwareReward，不将成本反馈给 policy**。

### 下一阶段接口

AI Judge 可离线读取 question/final_answer 与清洗后的 evidence，另存 judge 版本、评分和解释，再与此处 accuracy 对照，避免评分回流影响本阶段 policy。GRPO 前需要新增完整 prompt/completion token 对齐、assistant/action token mask、旧策略 logprobs、group ID 与终止 mask；明确 repair token、无效 action、截断和工具 observation 哪些进入损失。还需确定 reward 标度与组内归一化、数据划分和 CUDA 批量 rollout。此实现不包含这些训练功能。

### 本机验证与 prompt 修订记录

本机采用 CPU/float32 跑极小真实模型 smoke，正式 12×4 CUDA 实验仍需在 CUDA 环境运行。最初的 prompt 只在用户消息给出参数说明，8 条真实采样均因 SQL/参数层级错误而 invalid_action；这些原始输出与原始 prompt 完整保存在 `data/cpu_smoke_initial_prompt/`，报告为 `reports/base_policy_analysis_initial_prompt.md`。随后在 system prompt 中补全所有 action 的显式参数模板、query 对象嵌套与数值类型说明，再用相同任务和 seeds 跑最终 CPU smoke。两轮数据不混合统计；这也是一次 prompt 修订，不能将其变化解释成 RL 改进。

最终 CPU 报告为 `reports/base_policy_analysis_cpu_smoke.md`。后续正式实验应冻结此 prompt，并报告它对工具选择的潜在影响。所有实际执行和提出但因格式无效而未执行的 action 分开统计。

## Environment v1（独立兼容入口）

v1 实现位于 `src/env_v1/`，保留原环境、旧数据、奖励和训练接口。
30 students / 12 courses / 90 enrollments / 5 scholarships；40 题分布为 L1=10、L2=15、L3=15。
L3 每题复制独立数据库，五个奖学金各搭配三种已有申请状态。完整设计和实际结果见仓库根目录 `ENV_V1_PLAN.md`、`ENV_V1_RESULTS.md`。

```bash
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_env_v1.py
# 可选：在相同真实模型上复跑失败注入，输出必须分开
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_env_v1.py --failure-rate 0.2 --output results/env_v1_failure02.jsonl
```

baseline 使用已有本地 Qwen2.5-1.5B-Instruct 固定 revision、CPU float32、greedy decoding、8 次工具调用/9 次决策上限；无需新模型。CPU 全集耗时较长。每条完成即写入 JSONL；同步保存 KB、任务（含状态断言）、汇总及配置/数据/prompt 哈希。重跑相同路径会覆盖已有输出。

v1 search 是完整单表检索；lookup 返回完整实体；aggregate 支持 count/average/max/list；deep_search 支持奖学金资格或 year/min_gpa/course_id 关系条件；verify/submit 接受 student_ids 数组和 scholarship_id。成本分别为 1/1/1/5/1/2，direct_answer 为 0。一次批量提交成本为 2，逐项验证和拒绝错误；不自动替模型筛选待提交人选。

timeout 不执行工具；partial_result 执行后移除关键字段或最后一条结果（显式 partial_result=true），因此提交响应不完整时应先查状态再重试。每 episode 按 seed/task_id 固定注入序列。`failure_profile` 当前默认为 null，实验通过 runner 的 failure-rate 配置整体概率。

`task_success` 检查正确答案/完整最终状态且必须正常结束；被拒绝的重复/不合格尝试单独计入 violation，不抹去正确的最终状态。非法数据库变化强制失败。`trajectory_length` 包含最终 direct_answer，`tool_calls` 不包含；parse repair 次数见轨迹 parse_status。`recovery_success` 是有注入事件 episode 的最终成功，无注入为 null。未完成、解析错误和预算截断均计入成功率分母。该环境接口不返回训练张量；GRPO 由独立训练模块实现。
