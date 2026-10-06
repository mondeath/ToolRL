# Model Scale Diagnostic

固定 15 题，L1/L2/L3 各 5 题；任务 ID、权重 revision 和行为源码哈希保存在 configs/env_v1_diagnostic_tasks.json 与 configs/env_v1_model_scale.json。

| Model | Overall | L1 | L2 | L3 | Max-tool-call rate | Repeated actions | Avg cost | Deep search calls | Status / runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen2.5-1.5B-Instruct | 6.7% | 20.0% | 0.0% | 0.0% | 73.3% | 78 | 27.800 | 80 | 已复用历史结果; cpu/float32 |
| Qwen/Qwen2.5-3B-Instruct | — | — | — | — | — | — | — | — | 待云端运行 |
| Qwen/Qwen2.5-7B-Instruct | — | — | — | — | — | — | — | — | 待云端运行 |

Repeated actions = 每条轨迹内相邻、完全相同的已执行 tool name + arguments 的额外调用数之和；JSON 键顺序不影响判定。例如 A,A,A 计 2 次，A,B,A 计 0 次。包含执行后失败的调用，不含解析失败或预算阻止的提案。

`answer_ready_but_continued` 统一为 null：本阶段不严格推断 observation 是否足以构造答案，仅使用上述连续重复指标；null 不是 false。

统一 failure_rate=0、do_sample=false、max_tool_calls=8、max_steps=9、max_new_tokens=256、max_context_tokens=16384、seed=2026、float32；没有更改环境、task、tool schema、system prompt、evaluator 或 reward。

## 当前结论

证据尚不足以区分 1.5B 的 agent capability floor 与环境/protocol 问题。待三个模型在相同云端设备、精度和依赖版本下完成后再对照；CPU 归档不能当作完全相同 CUDA 配置的结果。

## 云端复现

实际 Git 根目录是 experiment/。下面命令从 clone 后的仓库根目录执行；若使用当前 ToolRL 工作区，先 cd experiment。详见 [MODEL_SCALE_RUNBOOK.md](MODEL_SCALE_RUNBOOK.md)。

```bash
python -m pip install -r requirements-inference.lock.txt -r requirements-test.txt
pytest -q
python scripts/run_model_scale.py --download-only
python scripts/run_model_scale.py --device cuda --output-root results/model_scale_cuda
```

不同输出根目录用于保留已归档 CPU 结果；每个模型目录仍包含 results.jsonl、summary.json、run.log。正式对照统一在 CUDA 上重新跑三个模型，无逐模型调参。
