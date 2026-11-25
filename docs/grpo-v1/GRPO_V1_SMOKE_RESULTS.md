# GRPO v1 smoke：实现与CPU验证结果

状态：IMPLEMENTED / CPU_VERIFIED / GPU_NOT_RUN。代码、测试和运行说明已完成；尚未加载7B或启动GPU训练，不存在GRPO后7B性能或显存实测。

## 已完成

- 固定24个train-only tasks：seeds1000–1007，L1/L2/L3各8题；配置见configs/grpo_v1_smoke_tasks.json。
- 原多轮rollout/evaluator完整复用，G=4、0/1 task_success reward、generated-token mask、group-relative advantage、adapter-only分段GRPO及最多3次更新。
- 7B NF4/bf16/LoRA加载、只读preflight、原15题greedy前后评估、adapter checkpoint保存/重载接口。
- 原Environment、数据生成器、schema、工具、业务规则、evaluator、固定diagnostic内容/hash均保持冻结。
- 全量单元测试：87 passed；结果保存在results/grpo_v1_smoke/verification/unit_tests.txt。最终GRPO专项复核见grpo_tests_final.txt。

## CPU合成验证（不是7B结果）

使用随机tiny Qwen2 + PEFT0.17.1真实LoRA，不下载模型、不使用GPU。固定合成token和reward仅用于数值测试，不进入正式环境训练。

| 指标 | 实测 |
|---|---:|
| total params | 23,712 |
| trainable params | 896 |
| trainable ratio | 3.7787% |
| optimizer steps | 1 |
| grad norm | 0.17016478 |
| adapter delta L2 | 0.000195888 |
| base权重hash不变 | 是 |
| checkpoint重载logprob一致 | 是 |

on-policy首次update的policy_loss为0，但梯度和adapter变化均非零；这是ratio=1且group advantage均值为0时的正常现象。测试还验证全0/全1组不执行optimizer.step，避免Adam动量造成假更新。

原位checkpoint重载及独立tiny base + PEFT重新加载均通过测试。真实7B/NF4 checkpoint重载仍需GPU运行后验证。

## preflight结果

静态检查passed：24题/8seed、diagnostic15题、train/dev/diagnostic隔离、冻结契约全部通过。

本机GPU preflight blocked（明确未开始训练）：

- CUDA不可用。
- bitsandbytes未安装。
- 未提供可读的本地7B snapshot。

本机torch2.8.0、transformers4.57.6、accelerate1.10.1；PEFT0.17.1仅安装在临时测试目录。未修改项目现有依赖环境。

7B实际total/trainable params、trainable ratio、GPU allocated/peak memory均待加载测量，不能用上述tiny数值替代。既有bf16 Base7B15题结果为11/15（73.33%），保留原产物；NF4 before/after均未运行，无本次性能结论。

## 产物与下一步

运行说明：GRPO_V1_IMPLEMENTATION.md。

验证产物：results/grpo_v1_smoke/verification/，包括static.json、gpu_preflight.json、cpu_synthetic/summary.json、unit_tests.txt与grpo_tests_final.txt。合成checkpoint位于忽略的checkpoints目录，未提交Git。

GPU实际运行须在用户审阅后进行：先在目标机器通过preflight，再显式--train；固定24组、每组4条轨迹，最多3次optimizer update。全部等reward时标记no_effective_update，不修改reward、环境或数据制造信号。
