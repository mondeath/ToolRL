# Environment v1 数据生成扩展

已实现并生成，未启动模型训练。新版入口为 `src.env_v1.generation.generate_instance(seed, split)`；旧 `src.env_v1.generate_kb` 保留用于冻结实验。新增模块不改变现有源码、schema、工具、业务规则、evaluator 或模型规模诊断配置。

## 生成方式

固定学生、课程、奖学金 ID，随机化 GPA、年级、部门、选课、完成状态、成绩、奖学金门槛/年级/必修课和初始申请。课程 credits 也在原数值字段中变化，使原有 count 任务的答案能够变化。每个 seed 保持 30 名学生、12 门课程、90 条选课、5 个奖学金。

五个六人 cohort 各自使用一门不同的核心课；补充课程从其余七门选择，防止无意跨 cohort 满足资格。每个奖学金恰有三名 eligible 学生，另有低 GPA、未完成核心课、年级不符三个失败角色。角色对应学生、数值和关系均随 seed 变化，不只是实体重命名。每个 KB 各奖学金有一或两名已申请者；原 task factory 为每个 L3 实例设置一名 eligible 已申请者、两名待申请者，与冻结任务结构一致。

任务字段集合及定义保留：L1 单表查询，L2 奖学金资格或年级/GPA/已完成课程交集，L3 合法提交全部待申请者并保留旧记录。L2 的固定 GPA 阈值仍为 3.5/3.65，年级和课程绑定新 cohort，并保留两个阈值分别筛出3/2人的结构。每个 seed 仍为 L1/L2/L3 = 10/15/15，共40题。

## 生成与审计结果

| split | seeds | KB 数 | task 数 |
|---|---|---:|---:|
| train | 1000–1029 | 30 | 1200 |
| dev | 2000–2009 | 10 | 400 |
| diagnostic（仅检查原数据） | 42 | 1 | 40 |

固定15题 diagnostic subset及其顺序完全保留；审计额外覆盖seed42全部40题以保护原eval数据。没有重写diagnostic产物。

- 41 个 KB hash 全部不同，41 个不含 ID/名字/引用的标量多重集签名也全部不同。因此不是单纯 ID permutation。
- 所有要求变化的字段均有变化：40 个新 KB 的 GPA、year、department、选课、completion、min_gpa、required_course、initial applications 向量各有40种；eligible_year 向量有37种。
- L1/L2/L3 的完整答案或 final-state 向量分别有41种；排除无关申请记录后，L3 自身 eligible/已申请/待申请目标集合也有41种。
- 全部1640条reference path通过原evaluator，success=true、rule violations=0。L1工具调用为1，L2为3，L3为5，direct_answer另计，保持原范围。
- train/dev/diagnostic 的 seed、task ID 和实例指纹均无交集；全部1640个实例指纹唯一。
- 原 `run_model_scale.load_contract()` 全部通过，包括冻结 KB、tasks、prompt、schema 和行为源码哈希。

“实例”指去掉 task_id 后的完整任务规范及其实际初始 KB，不使用 seed 标签制造不同哈希。相同任务模板、个别标量答案、课程数量或部分子问题可能跨世界重合；此检查不声称消除所有子问题/答案复用。每个完整世界及其无 ID 的事实签名均独立，符合本次同分布 seed 实例隔离要求。

## 产物及使用

`results/grpo_v1_data/` 包含 train/dev 各 seed 的 `kb.json` 和 `tasks.json`，以及 `manifest.json`（逐实例哈希、reference结果和来源）和 `summary.json`。该目录是独立数据池，不是已开始的1200题训练实验；后续GRPO smoke仍只选20–30个train task，dev仅评估。

从 experiment/ 运行：

```bash
python scripts/generate_grpo_v1_data.py --output-root results/grpo_v1_data_new
python -m pytest -q tests/test_env_v1_generation.py
```

生成器严格拒绝错误split或未分配seed，包括train使用42、dev使用1000。输出目录若存在则拒绝覆盖。相同seed重复生成完全相同。

新增测试覆盖：原冻结契约、schema与外键、全部seed可解性/工具步数、eligibility与申请结构、事实/答案变化、无ID签名、实例指纹不受task ID影响、split拒绝与重复检测、determinism、episode reset及输入不被修改。

最终全量回归：`python -m pytest -q` → **68 passed in 26.26s**（包括7个新增生成器测试）。最终版本的数据生成审计也独立运行通过。源码编译检查通过；本机默认系统字节码缓存目录不可写，编译检查使用 `/private/tmp` 缓存目录。

数据结构可解性不等于模型实际难度完全一致：这里保证原任务类别、角色约束和reference工具步数范围，模型性能留待后续独立评估。
