# GRPO v1：实现前设计与审阅门槛

状态：已完成仓库检查及用户授权的数据生成扩展（见 GRPO_V1_GENERATOR_RESULTS.md）；现已实现 trainer/collector、完成CPU测试和合成checkpoint验证；尚未运行7B GPU训练。当前可执行接口与验证见 GRPO_V1_IMPLEMENTATION.md。本文件保留设计决策。Environment v1、schema、prompt、parser/normalizer、evaluator 和 model-scale 结果保持冻结。

## 1. 当前交互与可复用组件

实际 Git 根为 experiment/。复用 src.rollouts.rollout、RolloutConfig、sampling_seed，src.env_v1.Environment、tasks.initial_state、evaluation.evaluate，以及 src.tasks.public_task。现有 src.rewards 的 MockAIReward / CostAwareReward 不适用于本 baseline，完全不调用。

build_messages 每次返回 system + user 两个 message；user 中包含 question、action schema、完整 JSON history 和下一步指令。它不是不断追加 assistant/tool messages 的聊天记录。工具成功后 history 加入 parsed action + observation；解析失败后加入原始输出 + format_feedback，最多 repair 一次。正常 action 的历史文本可能经过 normalizer，不能拿历史重分词恢复原始生成 token。direct_answer、被预算阻止的 action、失败解析输出也确实由模型生成。

保持 max_steps=9、max_tool_calls=8、max_parse_retries=1、max_new_tokens=256、max_context_tokens=16384、failure_rate=0。超 context 仍按现有逻辑终止，不截历史、不改任务。

## 2. token 对齐契约

新增 training policy 包装器，在每次 act 调用内部记录该次真实 build_messages、chat-template 后的 prompt_token_ids、模型直接返回的 generated_token_ids（含实际 EOS，不含 padding），以及 raw_output、generation index、session seed。返回既有 PolicyOutput，不改 rollout。

每次生成是独立训练 segment：input_ids = prompt_ids + generated_ids；mask = [0]*len(prompt_ids) + [1]*len(generated_ids)。用 logits[:, :-1] 预测 input_ids[:, 1:]，对应 mask[1:]。只提取 generated 部分 logprob，prompt 中的 observations、格式反馈及历史 action 副本一律 mask=0。此前生成 token 在其原始 segment 中训练一次，不能因后续历史重复计入 loss。

每条完整轨迹包含全部 segment（包括 repair 和最终无效/被预算拒绝的提案），共享一个 evaluator task-success reward 和 advantage。空生成/推理失败保留诊断但不伪造 token；基础设施错误停止训练，不能静默当作可学习失败。

保存 task_id/group_id/sample_id、每次调用 messages、actions、observations、termination_reason、success/reward、tool_cost/rule_violations、generated_token_ids/mask、old_logprobs、trajectory_length（沿用 evaluator）及 generated_token_count。训练对齐字段另存 training record，冻结 Trajectory 类型无需修改。

## 3. grouped rollout 与数据隔离：生成扩展已完成

每组 G=4，同一 public task 和同一 initial_state；逐条新建 env，复用不可变模型/tokenizer，每条独立 session seed，无共享 history/KV/state。独立采样不等于输出必定不同，不对重复输出强行重采样。训练 do_sample=true、temperature=0.8、top_p=0.95，eval do_sample=false。

用户已授权改变数据事实生成，保持环境规则、schema、任务定义及工具步数范围。新增 src/env_v1/generation.py 的 generate_instance(seed, split)，原冻结生成器与seed42不变。新入口变化GPA、year、department、选课/完成、奖学金门槛/年级/核心课、初始申请，不用ID重命名代替事实变化。

已生成train seeds1000–1029（1200题）、dev seeds2000–2009（400题），每seed40题。seed42全部40题仅作隔离及回归审计，固定15题diagnostic仍保留。全部1640条reference path成功，L1/L2/L3工具调用为1/3/5；41个KB及无ID事实签名各不重复，各层答案/最终状态向量确有变化。产物在results/grpo_v1_data/，详见GRPO_V1_GENERATOR_RESULTS.md。

后续smoke只从train池选24题（各层8题），不因生成了1200题而扩大本次训练。dev、diagnostic禁止进入训练。实例指纹排除task ID，纳入完整任务及实际初始KB；个别答案/模板重合不等同于整个实例重复，审计不宣称消除子问题复用。

## 4. loss 与采样概率

r_i = float(evaluate(task, env)['success'])，不添加任何成本/违规/长度项。
A_i=(r_i-mean(r))/(std_population(r)+1e-8)。std=0 时明确返回全0；统计 mixed_reward_group_rate、all_success_group_rate、all_failure_group_rate。全部相等的组不产生 policy gradient，跳过 optimizer.step，避免 Adam 动量/weight decay 造成无信号更新；分别记录 attempted_groups 与 optimizer_steps，不无限找混合组。

采用 token clipped GRPO：L = -(1/G) Σ_i [1/T_i Σ_{所有生成token} min(ratio*A_i, clip(ratio,1-0.2,1+0.2)*A_i)]。T_i 是整条轨迹生成 token 数，不能对每个 turn 等权，否则短 action 权重偏大。逐 segment backward，按 G*T_i 归一化，整组最后统一 clip_grad_norm 和 step。old_logprobs 在 rollout 时/更新前无梯度计算并固定；一组一次 update，禁止收集途中更新。

第一版明确采用通常的原始模型 softmax logprob 作 policy surrogate，old/current 都不套 temperature/top-p；temperature/top-p 只用于探索。这不是截断采样行为分布的严格 importance ratio，文档与产物必须标明该近似，不能把 generate 返回的 warped scores 混作 old logprob。对齐测试验证 update 前 ratio=1。关闭 adapter dropout，rollout/update 使用一致的 dropout 行为。

## 5. reference / KL

首版 beta=0，KL 不进入 loss、不需要 reference 或第二套7B权重；日志 KL=null（disabled），不是测得0。old policy 用冻结的逐 token logprob 表示，不需要旧模型副本；没有 value head/critic。

若后续明确启用 KL，可以同一 NF4 base 临时 disable_adapter 获取无 adapter reference，no_grad 后重新启用；参考分布是量化 base，不是完整 bf16 base。可用 exp(logp_ref-logp)-1-(logp_ref-logp) 逐生成 token 估计，届时必须单独验证并记录 beta。首版不实现额外 KL 路径。

## 6. trainer 选择

使用小型自定义分段 GRPO trainer。不是声称 TRL 不支持多轮：当前 TRL 有 rollout_func / environment_factory，但默认一条 completion 的对齐和工具模板不能直接表达当前每轮重建 system+user、将 normalizer 后 action 写入 user history 的语义。适配需要专门分段 probability 契约，直接使用工具循环会改交互；本机也没有安装 TRL，现有 transformers=4.57.6 与当前环境接口的版本要求不同。不为 trainer 升级冻结推理栈或改协议。

参考：[TRL 文档](https://huggingface.co/docs/trl/grpo_trainer)、[官方实现](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)、[PEFT 量化](https://huggingface.co/docs/peft/developer_guides/quantization)。依赖兼容以云端实际检查为准，不将主分支文档视为已锁定安装版本。

## 7. 7B / 单24GB 显存方案

固定 Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28。4-bit NF4 + double quant、bf16 compute、prepare_model_for_kbit_training、gradient checkpointing（训练 use_cache=false）。LoRA r=16、alpha=32、dropout=0；候选 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj；在实际 named_modules 上逐项确认再加载，不默默忽略缺失模块。只训练 adapter，AdamW weight_decay=0。

顺序4条采样，generation batch=1，backward segment batch=1，CPU 保存 tokens/logprobs。生成阶段 no_grad/use_cache=true，生成后释放 KV；训练阶段 checkpoint + SDPA，不能同时保留所有 turn 的图。尽可能仅对生成位置做 vocab projection / 分块计算 logprob，避免16384×152064完整 fp32 logits（单份约9.3GiB）；需要与标准 forward 数值/梯度等价测试。

预算是估计而非实测：量化 base 及未量化 embedding/head/metadata 约5–7GiB（prepare 可能把非量化参数转fp32，应核对）；adapter/grad/optimizer 约0.5–1GiB；激活及分块 logits 约3–8GiB；KV/allocator/workspace 约2–4GiB。预计常见长度11–20GiB，但16k最坏情况不保证24GB可运行。OOM时停止并记录，优先调整 projection chunk/checkpoint 实现，不降低任务预算或换模型。

实际加载后输出 trainable_parameter_count、logical_total_parameter_count（量化storage numel不能冒充逻辑参数总数）、trainable_ratio 和 storage bytes；预计全七类投影 r16 约40M adapter参数，仅作预算。记录 allocated/reserved、max_memory_allocated/max_memory_reserved、GPU型号/容量及每阶段 peak。无CUDA时不能报告实测peak。

本地检查：Python3.9.6、torch2.8.0、transformers4.57.6、accelerate1.10.1；peft/bitsandbytes/trl 未安装，CUDA不可用。云端 preflight 必須检查CUDA/bf16、驱动、Python/wheel兼容、NF4 forward/backward、上述模块和冻结参数；成功后另存训练依赖lock，不改 requirements-inference.lock.txt。不擅自安装/切模型。

## 8. 实现结构与测试

- src/env_v1/generation.py、src/evaluation/generator_audit.py：已实现的数据实例化和泄漏审计；训练data loader已从隔离数据池固定选择24题。
- policy.py：QLoRA加载、act捕获、session隔离，保持原Policy边界。
- collector.py：原rollout + 原evaluate，组记录和逐call segment。
- loss.py：shift/mask、logprob、advantage、全轨迹归一化。
- trainer.py：有限update loop、adapter-only optimizer、metrics/checkpoint。
- scripts/run_grpo_v1_smoke.py：prepare / preflight / train / eval / report 显式阶段。
- configs/grpo_v1_smoke.json、requirements-training.txt、tests/test_grpo_v1_*.py。

必测：组内同task；独立seed/session/env且允许相同输出；evaluator success是唯一reward来源（改变cost/诊断违规不额外改reward）；observation和历史action mask=0；repair/EOS/shift/mask长度一致；all-equal advantage=0；混合组更新adapter；base逐参数冻结且值不变；checkpoint reload后的logprob一致；重复episode state reset；diagnostic/eval不进train；全轨迹分段梯度等价；无信号组不step。tiny本地模型可验证算法，但不能替代真实7B NF4测试。

## 9. smoke与评估门槛

数据生成决策与数据审计已完成；接下来完成trainer及CPU测试，再给用户review，确认后才启动GPU。固定24 tasks、G4、单轮最多24 attempted groups、最多3 optimizer updates；没有混合组则报告无更新，不能宣称adapter更新通过。先一组NF4内存/反传探针，再有限smoke。保存adapter、tokenizer、config、base revision、optimizer/RNG state、训练数据manifest和依赖版本；重启加载验证后greedy评估。checkpoint目录加入Git忽略，不覆盖已有结果目录。

训练前/后均跑固定15题，保持原顺序和greedy配置。保留已冻结bf16 Base7B结果；另外跑NF4 adapter-off baseline和NF4 smoke checkpoint同精度比较，避免量化差异冒充训练效果。各自输出overall/L1/L2/L3 success、invalid_action/max_tool_call rate、相邻重复动作、avg cost、success & zero violation。smoke不作为性能结论。

日志：step/attempted_groups、mean_reward/task_success、三种group率、policy_loss、KL=null、gradient_norm、learning_rate、trajectory_length/generated_token_count、invalid_action/max_tool_call率、cost/violations、GPU allocated/reserved/peak、adapter_delta、trainable/total/ratio。group率覆盖所有已采样组。

## 10. 已实现命令与GPU门槛

实际入口见GRPO_V1_IMPLEMENTATION.md。旧的prepare/max-updates接口已收敛为显式static/preflight/cpu-smoke/train/eval模式；update上限由已校验config固定，不支持长训覆盖参数。

```bash
python scripts/run_grpo_v1_smoke.py --static-check
python scripts/run_grpo_v1_smoke.py --preflight --model-dir "$MODEL_DIR"
python -m pytest -q
python scripts/run_grpo_v1_smoke.py --cpu-smoke --output-root results/grpo_v1_smoke/cpu_check_new
# 以下须在GPU smoke获准后执行，本次未执行：
python scripts/run_grpo_v1_smoke.py --train --model-dir "$MODEL_DIR" --output-root results/grpo_v1_smoke/run_001
python scripts/run_grpo_v1_smoke.py --eval --model-dir "$MODEL_DIR" --checkpoint results/grpo_v1_smoke/run_001/checkpoints/adapter --output-root results/grpo_v1_smoke/reload_eval_001
```

数据生成器已冻结。固定24题来自8个train-only seeds；本次只做代码/单元测试/CPU合成验证，GPU实测参数量、显存和15题训练前后结果仍待获准后的运行。
