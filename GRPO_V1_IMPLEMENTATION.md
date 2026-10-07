# GRPO v1 smoke 实现与运行说明

已实现固定选题、分组多轮采样、纯 task-success reward、generated-token GRPO loss、QLoRA 更新、checkpoint 和 greedy 前后评估。当前只运行 CPU 单元测试、静态检查和 tiny Qwen + **真实 PEFT LoRA** 合成 loss smoke；**没有加载或训练 7B，没有 GPU 实测结果**。

## 固定数据及冻结边界

`configs/grpo_v1_smoke_tasks.json` 按固定顺序选择24题，L1/L2/L3各8题；来自train seeds1000–1007，每个world各选三种难度各1题。选择与模型采样结果无关，不动态换题。

`configs/grpo_v1_smoke.json` 固定任务列表hash、生成器源码hash、原数据manifest hash、diagnostic列表hash。加载时验证原model-scale冻结源码/KB/tasks/prompt/schema，再检查各选中world的实际KB/tasks hash，实例指纹排除task ID，核对全部dev与seed42原40题的实例和ID不重合。固定15题diagnostic顺序保持原样。没有修改生成器、schema、环境、工具行为、解析器或evaluator。

## 代码入口

| 文件 | 职责 |
|---|---|
| `src/training/grpo_v1/data.py` | 固定选题、冻结契约、train/dev/diagnostic隔离 |
| `policy.py` | 单卡NF4/PEFT加载，复用原HuggingFacePolicy.act并捕获真实token |
| `collector.py` | 原rollout + 原evaluate；每题4条新env/session |
| `loss.py` | generated-token对齐、group advantage、clipped GRPO |
| `trainer.py` | adapter-only更新、base hash、checkpoint保存与重载 |
| `evaluation.py` | 原15题greedy和诊断统计 |
| `preflight.py` | 只读环境/模型路径/数据检查，不加载7B权重 |
| `synthetic.py` | 显式CPU测试入口，随机tiny Qwen；不替换正式主模型 |
| `scripts/run_grpo_v1_smoke.py` | 必须显式选择static/preflight/cpu-smoke/train/eval |

## multi-turn与token mask

完全复用现有rollout：每轮仍由 `build_messages` 构造system+user，user内含question、schema和完整history；action→env observation→下一轮action，直到direct_answer/原终止条件。不是把环境改成单completion，也没有改成另一套assistant/tool模板。

每次真实generation保存一个segment：messages、prompt_token_ids、generated_token_ids（含实际EOS）、generated_token_mask、raw_output、generation_index、session_seed、old_logprobs。mask为`[0]*prompt_length + [1]*generated_length`。

loss的预测位置为`prompt_length-1 ... sequence_length-2`，目标为真实generated IDs；Qwen的`logits_to_keep`仅投影这些位置，避免为全部长prompt分配vocab logits。系统指令、用户内容、所有observation/format feedback、历史中复制的旧action均不作为target。repair、无法解析输出和预算阻止的生成也按其原始token训练；不能用normalizer后的文本重建target。

每组4条episode都新建Environment(initial_state)，沿用原sampling_seed/new_session，history和KV不共享。采样结果允许相同，不能为了制造不同结果重采样。基础设施错误停止运行；context_limit保持原终止语义，不截历史。

## reward、advantage和loss

唯一reward为 `float(original_evaluator_result['task_success'])`。cost、violation、verifier、长度、失败注入均不进入reward。若evaluator本身因非法状态判失败，仍照原结果取0。

使用总体标准差：`A_i=(r_i-mean(r))/(std_population(r)+1e-8)`。全0/全1明确返回全0 advantage，并**完全跳过optimizer.step**，避免Adam动量继续改参数。mixed/all-success/all-failure rates按所有已采样group累计计算。

`L = -(1/G) Σ_i (1/T_i) Σ_generated_tokens min(ratio*A_i, clip(ratio, 0.8, 1.2)*A_i)`。

T_i为整个trajectory的生成token总数，包括全部turn/repair。逐segment backward，最后整组clip_grad_norm和一次step；不是各turn等权。old logprob在采样时无梯度计算，整组采集期间不更新。每组最多一次update，不重复使用旧group。

训练采样do_sample=true、temperature=0.8、top_p=0.95，并显式top_k=0；greedy eval do_sample=false。old/current均为**原始模型softmax logprob**，不使用generate中经temperature/top-p变换的分数；该surrogate不是截断行为采样分布的严格importance correction，保留设计中声明的近似。

beta=0；KL日志为null（disabled），没有reference模型，也没有value head/critic。adapter on/off只用于获取同一个NF4 base的训练前greedy基线，不用于KL。无需第二份7B权重。

## 24GB QLoRA方案

主模型固定Qwen/Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`。仅从已有本地HF snapshot读取，无下载、CPU fallback或模型替换。

- NF4 4-bit + double quant，bf16 compute，SDPA；全部模型放CUDA:0，不使用自动CPU offload。
- LoRA r=16、alpha=32、dropout=0、bias=none；q/k/v/o/gate/up/down投影。加载时验证每类都有28个module。
- prepare_model_for_kbit_training + non-reentrant gradient checkpointing；训练use_cache=false，生成no_grad/use_cache=true。
- k-bit准备后将冻结的embedding/lm_head恢复bf16，norm保留fp32；生成和logprob重放都使用bf16 autocast，避免无意使用fp32的hidden/KV缓存。
- optimizer只包含LoRA A/B参数，base.requires_grad=false；AdamW，lr=1e-5，weight_decay=0，grad norm上限1。
- sequential rollout batch1、segment backward batch1、CPU保存旧logprob；不同时保留多个turn的计算图。
- selective logits最多256个生成位置，避免16384×vocab完整logits；单独测试与完整投影的值及梯度等价。

参数总数按4-bit quant_state的逻辑shape统计，而不是uint8打包storage numel。加载后输出total_params、trainable_params、trainable_ratio；parameter_storage_bytes仅参数存储，不包括量化metadata。另输出真实GPU allocated/reserved/peak。全七类r16 adapter预计约40.37M参数，最终以加载输出为准。

显存设计预算约11–20GiB（依实际prompt长度和allocator而异），**不保证16k最坏情形能在24GB运行**。OOM会停止并留下错误状态，不能自动缩工具预算、换模型或裁掉history。

## preflight与依赖

`requirements-training.txt` 与原推理lock分开；PEFT=0.17.1、bitsandbytes=0.48.1、Transformers4.57.x。训练依赖需在目标CUDA机器安装；本次只在临时目录安装了PEFT来运行真实CPU LoRA测试，没有改变已有项目环境或安装GPU依赖。

preflight检查：冻结契约、24题选择和split隔离、torch/transformers/accelerate/PEFT/bitsandbytes能导入、指定PEFT/bnb版本、CUDA available、bf16 support、GPU name/total memory（至少22GiB可见显存，兼容标称24GB设备的保留空间）、Qwen selective-logits API、本地snapshot的config/tokenizer/generation config和所有safetensors shards可读。

要求标准`models--Qwen--Qwen2.5-7B-Instruct/snapshots/<revision>`路径并核对7B架构。路径和架构校验不是权重内容的完整checksum，preflight不会读取全部7B权重或声称验证NF4 CUDA kernel。NF4实际加载/反传及显存峰值仍须在获准后的GPU smoke验证。依赖导入成功也不等于kernel兼容成功。

## 可执行命令

以下命令从experiment/执行。已有输出目录会拒绝覆盖。程序没有默认train模式，`--preflight`永远不会自动接着训练。

```bash
# CPU/静态；无需GPU和模型snapshot
python scripts/run_grpo_v1_smoke.py --static-check
python -m pytest -q

# 需要torch/transformers/PEFT，但无须下载模型；随机tiny Qwen，真实LoRA
python scripts/run_grpo_v1_smoke.py --cpu-smoke \
  --output-root results/grpo_v1_smoke/cpu_check_002

# 目标GPU机器：依赖安装后只做只读preflight
python -m pip install -r requirements-training.txt -r requirements-test.txt
MODEL_DIR=/root/shared-nvme/hf-cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
python scripts/run_grpo_v1_smoke.py --preflight --model-dir "$MODEL_DIR" \
  --output-root results/grpo_v1_smoke/gpu_preflight_001
```

下面是**用户审阅并确认GPU smoke之后**才运行的命令，本次未执行：

```bash
python scripts/run_grpo_v1_smoke.py --train --model-dir "$MODEL_DIR" \
  --output-root results/grpo_v1_smoke/run_001

# 独立进程从checkpoint重建单份量化base+adapter，再跑greedy eval
python scripts/run_grpo_v1_smoke.py --eval --model-dir "$MODEL_DIR" \
  --checkpoint results/grpo_v1_smoke/run_001/checkpoints/adapter \
  --output-root results/grpo_v1_smoke/reload_eval_001
```

train固定遍历24题×4rollouts一次，最多3次optimizer update；用满3次后剩余group仍采样和记录，但不更新。不多epoch、不反复找mixed group。全程无mixed信号则status=no_effective_update，不能报告adapter更新验证通过。

## checkpoint、评估与日志

训练前将现有固定15题 **bf16 Base7B** 的结果来源/hash和指标另存，保留原产物。本仓库原基线为11/15 success（73.33%），不是本次重新测得。

随后在单份NF4 base上disable_adapter跑15题greedy；训练后保存adapter并原位重载，验证生成token logprob一致，再跑同15题greedy。主要训练对照是NF4 before vs NF4 after，旧bf16结果单列，不把量化差异归因于训练。另提供独立进程reload eval命令。

checkpoint为PEFT adapter+tokenizer，附config/base revision、optimizer state、step、CPU/CUDA RNG及数据选择hash。没有合并base或保存第二份完整7B。当前不提供断点续训CLI，optimizer/RNG仅用于审计及后续扩展。所有checkpoints目录已由现有.gitignore排除。

每个group日志含step、attempted_groups、reward/success、三种group率、policy_loss、KL=null、grad_norm、lr、trajectory length、generated token count、invalid_action/max_tool_call率、cost/violations、GPU allocated/reserved/peak和adapter变化量。group率累计，reward和episode指标为当前group均值。

每条trajectory记录messages/actions/observations/termination/success/reward/cost/violations及逐segment的token IDs、mask、old logprob。eval汇总overall/L1/L2/L3 success、invalid_action、max_tool_call、相邻重复动作、avg cost、success & zero violation、合法action解析率。没有合法tool action的后评估会显式标为异常状态，不当作完全通过。

```text
run_001/
  config.json / smoke_tasks.json / preflight.json / runtime_requirements.txt
  archived_bf16_base.json / model_runtime.json
  before_nf4/results.jsonl + summary.json
  rollouts.jsonl / metrics.jsonl
  checkpoints/adapter/                         # Git ignored
  checkpoint_reload.json
  after_nf4/results.jsonl + summary.json
  summary.json / status.json
```

## 本机验证及剩余限制

全量回归87项通过，最终GRPO专项复核19项通过（无PEFT skip）；日志见verification目录。

已完成的CPU合成smoke使用23,712总参数、896可训练LoRA参数（3.7787%）；一次update的grad_norm约0.17016，adapter delta L2约0.00019589，base hash不变，checkpoint重载logprob一致。这些数值**仅属于tiny CPU模型，不能当作7B结果**。on-policy ratio初始为1且advantage均值为0，policy_loss可恰为0而梯度非零，测试验证了真实参数变化。

最终验证产物保存在 `results/grpo_v1_smoke/verification/`：`static.json`、`gpu_preflight.json`、`cpu_synthetic/summary.json`、`unit_tests.txt`。

测试覆盖固定24题与dev/diagnostic拒绝、组内同task、独立session和env reset、原success唯一reward、observation/repair/EOS mask、全0/全1 advantage及不执行Adam、分段与整轨迹梯度等价、selective/full logits值与梯度一致、adapter更新/base冻结、真实PEFT checkpoint原位及新模型重载、update budget和实际HF act捕获。

当前主机CUDA不可用、未安装bitsandbytes、未提供本地7B snapshot；GPU preflight明确返回blocked，不会尝试CPU跑7B。PEFT测试用临时目录路径 `/private/tmp/toolrl-grpo-peft`；复现实测测试可使用 `PYTHONPATH=/private/tmp/toolrl-grpo-peft ../.venv/bin/python -m pytest -q`。干净环境应安装PEFT后再跑，不能把PEFT测试被skip视为全部验证通过。

实现参考：[PEFT量化与k-bit训练](https://huggingface.co/docs/peft/developer_guides/quantization)。实际selective-logits接口以本机Transformers4.57.6源码及数值测试核对。
