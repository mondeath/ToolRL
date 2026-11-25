# Environment v1 最小改动方案

## 1. 当前环境结构
`experiment/src/environment/ToolEnvironment` 复制 KB，按 episode 记录调用和成本，direct_answer 结束；reset 仅接受 task_id/question。工具预算由环境和 rollout 双重约束。原环境 reset 不恢复数据库（原工具只读）。

## 2. 当前工具
lookup 单实体字段，search 单表过滤/关键词（top-k），deep_search 关系路径查询，direct_answer。现有成本 1/2/8/0。查询引擎可复用单表查询；跨列表条件为独立 any-match，不能直接用于课程完成状态的关联判断。

## 3. 当前 task schema
64 条旧任务包含 task_id/question/task_type/ground_truth/evaluation。public_task 只暴露 task_id/question，评估字段隔离。保留全部旧任务及测试；新增独立 40 条 v1 任务：10 L1、15 L2、15 L3。L2 不保存最终答案，只保存离线计算条件。

## 4. 当前 reward/verifier 接口
AIReward.score(question, final_answer, evidence)，CostReward.score(tool_calls)，PureReward/CostAwareReward；MockAIReward 不是正确性 verifier。当前无 eligibility verifier、状态评测或 failure injection 接口。v1 新增这些能力，保留奖励接口，不改训练逻辑。

## 5. 复用
复用 QueryEngine 单表 DSL、Trajectory、public_task、normalized_equal、HuggingFacePolicy、本地固定 revision Qwen2.5-1.5B-Instruct、rollout 的解析修复和采样/预算/日志。旧 schema/parser/prompt 默认行为不变。rollout 根据环境选择可选 schema/parser，build_messages 根据 v1 schema 选择独立 prompt。

## 6. 实施方案
- 独立 v1 模块，固定 seed 生成 30 students、12 courses、90 enrollments、5 scholarships 和少量 applications。
- 七个 action：search/lookup/aggregate/verify 成本 1，submit 成本 2，deep_search 成本 5，direct_answer 成本 0。search 单表；deep_search 结构化关系查询；submit 可批量且逐项验证，无效/重复拒绝并记录违规。
- L2 组合课程/学生/奖学金条件；L3 每题独立初始申请状态，包含多个候选、不合格、已申请和待申请。用 3–5 调用参考执行验证可达性，参考脚本不冒充 baseline。
- timeout 在执行前失败；partial_result 在执行后裁剪响应（写入仍可能已提交），固定 episode seed、可配置概率，记录注入事件。
- 离线评测答案及 explicit final_assertions/forbidden_assertions，验证非申请表不可变、原申请不可修改、无非法新增；分开记录成功/违规/成本/长度/昂贵调用/恢复。
- 新增测试覆盖工具、关系语义、隔离、失败、状态断言、成本、rollout 兼容；运行 pytest 和旧 smoke。
- 先跑 failure_rate=0 的 40 条真实 Qwen baseline，逐条保存 JSONL，再测试 0.2/0.5 的确定性失败。报告原始结果及典型轨迹，不按目标成功率修饰结果。
