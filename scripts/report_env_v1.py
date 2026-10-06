"""Render the Environment v1 report from completed, unmodified model results."""
from _common import ROOT
import json


def render(rows, summary):
    rates=summary['success_by_difficulty']
    pct=lambda x:f'{x:.1%}'
    from collections import Counter
    terminations=dict(Counter(r['trajectory']['termination_reason'] for r in rows))
    lines=['# Environment v1 结果', '', '## 改动与兼容', '',
        '新增 `src/env_v1/`，保留原环境、旧任务、奖励和训练逻辑。复用 QueryEngine、Trajectory、public_task、HuggingFacePolicy 与 rollout；rollout 仅新增环境可选 schema/parser，消息构造按 schema 版本选择 prompt。', '',
        '固定 KB seed=42：30 students、12 courses、90 enrollments、5 scholarships、5 初始 applications。课程完成与课程 ID 在同一 enrollment 上匹配。每个 episode 深拷贝并恢复初始状态。', '',
        '七个 action：search / lookup / aggregate / verify 成本 1，submit 成本 2，deep_search 成本 5，direct_answer 成本 0。submit 支持显式 ID 批次并逐项拒绝无效/不合格/重复申请；一次批次成本 2，使完整 L3 可在 5 次调用内完成。deep_search 是合法的昂贵关系查询快捷路径，不强制惩罚工具选择。', '',
        '旧项目无独立 verifier 和 failure injection，因此新增 eligibility 与最终状态检查，未替换已有奖励接口。search 使用完整的小表结果，无分页工具；L2 cheap path 经三次检索完成，deep_search 可以一次完成同一查询。此权衡是成本实验设计的一部分。', '',
        '## 任务与评测', '',
        '| 难度 | 数量 | 内容 | 参考工具调用数 |', '|---|---:|---|---:|',
        '| L1 | 10 | 5 GPA、4 year 列表、1 course count | 1 |',
        '| L2 | 15 | 5 scholarship 资格、10 course/year/GPA 组合 | 3 |',
        '| L3 | 15 | 5 scholarship × 3 独立已有申请状态 | 5 |', '',
        '每个 L3 有 6 个本课程候选：3 名符合资格（1 名已经申请、2 名待提交），以及 GPA 或完成状态不满足的候选。L3 的三个变体共享题目文本但初始申请者不同；它们是状态变体，不是 15 个独立任务模板。', '',
        '公开任务仍只有 task_id/question；任务文件的 evaluation 条件与 final_assertions 不进入 policy。L2 不保存最终答案，评测从初始 KB 计算；L3 显式比较完整 applications，并检查其他表和已有申请不可修改。参考路径只用于测试可达性，不参与 baseline 决策。', '',
        'task_success 要求最终答案/状态正确、正常 direct_answer 结束且无非法状态变化；拒绝成功的重复/不合格尝试另记 violation，因此 task_success 与零违规不等价。trajectory_length 包含 direct_answer，tool_calls 不包含；parse repair 另存 parse_status。recovery_success 仅针对有注入事件的 episode，表示其最终成功率；无注入为 null。', '',
        '## 真实 Qwen baseline', '',
        f'模型：Qwen2.5-1.5B-Instruct；revision `{summary["run"]["model_config"]["revision"]}`。CPU float32，greedy decoding（do_sample=false），每轮最多 256 新 token，最大上下文 16384，最多 8 次工具调用/9 次决策，每次最多一次格式修复。failure_rate=0，40 题各一次，未筛选成功轨迹。', '',
        f'运行耗时 {summary["run"]["elapsed_seconds"]:.1f} 秒。完整配置、运行版本、KB/task/prompt 哈希见 `results/env_v1_baseline.summary.json`。', '',
        '| 指标 | 实测 |','|---|---:|',
        f'| overall success | {pct(summary["overall_success"])} |',
        *[f'| {l} success | {pct(rates[l])} |' for l in ('L1','L2','L3')],
        f'| average tool cost | {summary["average_tool_cost"]:.3f} |',
        f'| average trajectory length | {summary["average_trajectory_length"]:.3f} |',
        f'| average tool calls | {summary["average_tool_calls"]:.3f} |',
        f'| deep_search usage（至少一次的 episode 比例） | {pct(summary["deep_search_usage"])} |',
        f'| deep_search calls | {summary["deep_search_calls"]} |',
        f'| rule violation rate（有违规的 episode 比例） | {pct(summary["rule_violation_rate"])} |',
        f'| rule violation count | {summary["rule_violation_count"]} |',
        '| recovery_success | null（本轮无注入） |', '',
        '## 模型失败分析', '',
        'termination 分布：`'+json.dumps(terminations,ensure_ascii=False)+'`。', '',
        '| 难度 | 平均成本 | 平均工具调用数 |', '|---|---:|---:|',
        *[f"| {level} | {sum(r['tool_cost'] for r in rows if r['difficulty']==level)/sum(r['difficulty']==level for r in rows):.3f} | {sum(r['tool_calls'] for r in rows if r['difficulty']==level)/sum(r['difficulty']==level for r in rows):.3f} |" for level in ('L1','L2','L3')], '',
        '工具实际调用分布：`'+json.dumps(dict(Counter(c['tool_name'] for r in rows for c in r['trajectory']['tool_calls'])),ensure_ascii=False)+'`。预算耗尽与解析失败均按失败计入分母。低违规率也可能只是未执行 submit，不能单独解释成模型能够安全完成申请。', '',
        '## 典型轨迹', '']
    # Prefer distinct difficulty/tool outcomes over adjacent near-identical tasks.
    def choose(success,count):
        candidates=[r for r in rows if r['success']==success]
        selected=[]
        signatures=set()
        for level in ('L1','L2','L3'):
            first=next((r for r in candidates if r['difficulty']==level),None)
            if first is not None:
                selected.append(first)
                signatures.add((first['difficulty'],tuple(c['tool_name'] for c in first['trajectory']['tool_calls']),first['trajectory']['termination_reason']))
                if len(selected)==count:return selected
        for r in candidates:
            signature=(r['difficulty'],tuple(c['tool_name'] for c in r['trajectory']['tool_calls']),r['trajectory']['termination_reason'])
            if signature not in signatures:
                selected.append(r);signatures.add(signature)
                if len(selected)==count:return selected
        for r in candidates:
            if r not in selected:selected.append(r)
            if len(selected)==count:break
        return selected
    for outcome,count in ((True,2),(False,3)):
        chosen=choose(outcome,count)
        if len(chosen)<count:lines += [f'实际仅有 {len(chosen)} 条可展示的{"成功" if outcome else "失败"}轨迹；不补造样本。','']
        for row in chosen:
            t=row['trajectory']
            lines += [f'### {"成功" if outcome else "失败"}：{row["task_id"]}', '',t['question'],'',
                f'cost={row["tool_cost"]}，tool_calls={row["tool_calls"]}，violations={row["rule_violation_count"]}，termination={t["termination_reason"]}。','', '```json',
                json.dumps(dict(tool_calls=t['tool_calls'],final_answer=t['final_answer'],raw_outputs=t['raw_outputs'],rule_violations=row['rule_violations']),ensure_ascii=False,indent=2),'```','']
    scalar_array_failures=[r['task_id'] for r in rows if not r['success'] and r['difficulty']=='L1' and isinstance(r['trajectory']['final_answer'],list) and len(r['trajectory']['final_answer'])==1]
    lines += ['## 验证与失败注入','',
        'pytest 的实际执行记录见 `results/env_v1_tests.txt`，原测试与新增测试全部通过。40 条测试专用参考路径（`results/env_v1_reference.jsonl`，明确标注非模型结果）全部成功且无违规，L2/L3 为 3–5 tool calls；这证明环境可解，不是模型性能。新增测试覆盖搜索/完整 lookup/聚合/关系查询/eligibility/重复与不合格提交/最终状态/非法修改/成本/reset 隔离/解析/rollout/timeout/partial_result/恢复。', '',
        '旧 smoke 在临时目录跑完 192 条轨迹，economical / redundant_deep 均为 100%，未覆盖原有数据；结果见 `results/env_v1_legacy_smoke.json`。', '',
        'timeout 执行前返回失败，不产生写入；partial_result 执行后裁剪结果并显式标记，写入可能已经成功。测试验证 partial submit 后先 verify 再决定重试。', '',
        '`results/env_v1_failure_smoke.json` 为固定 seed 的脚本 lookup 重试探针，分别以 0/0.2/0.5 概率跑 50 次；它不衡量 Qwen 的恢复能力。真实模型本轮只跑 failure_rate=0，后续可用同一 runner 的 --failure-rate 复跑。','',
        '## 是否适合进入 GRPO','']
    if scalar_array_failures:
        lines += [f'L1 出现 {len(scalar_array_failures)} 条单元素数组形式的失败（例如要求数值却输出 `[3.8]`）。评测保留既有严格 scalar/list 规则；没有为提高成功率调整结果。这提示协议遵从可能混入能力测量。','']
    if any(rates[l]<.2 for l in ('L2','L3')):
        lines += ['至少一个 L2/L3 成功率低于 20%，尚不满足直接进入 GRPO 的难度目标。需先分析协议错误、候选筛选和状态操作失败，冻结明确的标量/列表输出示例后另开实验验证，保留本轮结果。','']
    elif any(rates[l]>.85 for l in ('L2','L3')):
        lines += ['至少一个 L2/L3 成功率超过 85%，当前关系/状态变体对该 baseline 仍偏易。下一步应增加少量不对称候选与状态变体，再独立评测；不要按目标成功率改写结果。','']
    elif .5<=rates['L2']<=.8 and .3<=rates['L3']<=.7 and rates['L1']>=.8:
        lines += ['本轮落在目标参考区间，环境可作为下一阶段 GRPO 的起点；仍需独立验证集和多 seed 复测。','']
    else:
        lines += ['环境已具备成本与状态区分度，但本轮未同时落在 L1 80–100%、L2 50–80%、L3 30–70% 的目标区间；建议先排查协议与任务分布，再决定是否训练。','']
    lines += ['当前仅 40 条、单 seed、greedy baseline，且 L3 同奖学金任务共享结构；不能据此推断泛化性能。先固定 prompt 与评测规则，补真实模型 0.2/0.5 failure 实验及小规模独立种子验证，再讨论 reward 权重与 GRPO。没有实现 GRPO、引入新模型、user simulator 或大型框架。','',
        '## 复现','', '```bash', '.venv/bin/python -m pytest -q',
        'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiment/scripts/run_env_v1.py',
        '.venv/bin/python experiment/scripts/check_env_v1_failures.py',
        '.venv/bin/python experiment/scripts/report_env_v1.py','```','']
    return '\n'.join(lines)


def protocol_comparison(initial_rows, initial_summary, final_rows):
    from collections import Counter
    lines=['', '## 首轮协议诊断（独立保存，不混入最终指标）', '',
        f'首轮同样跑完 {len(initial_rows)} 题，overall success={initial_summary["overall_success"]:.1%}，L1/L2/L3='+
        '/'.join(f'{initial_summary["success_by_difficulty"][l]:.1%}' for l in ('L1','L2','L3'))+'。', '',
        '首轮使用简写参数示例，模型常把 deep_search 条件放到 arguments 顶层，漏掉 query；还出现标量返回单元素数组、反复执行同一查询。首轮未修改或删去，见 `results/env_v1_initial_protocol.jsonl` 与配套 summary/KB/tasks/log；提示词原文见 `results/env_v1_initial_protocol_prompt.txt`，已与首轮 prompt 哈希核对。', '',
        '正式重跑仅更新 v1 system prompt：改为每种 action 的完整 JSON 示例，明确 query 嵌套、标量/列表格式、不得混合两种关系查询格式，以及完成检索后结束而不重复确定性查询。没有改变任务、数据库、工具、评分规则、模型或生成参数。这是基于同一任务集做的协议调试，不能视作独立留出评估，也不是 RL 改进。最终 prompt/messages 保存于 `results/env_v1_baseline.prompt.json`。', '',
        '首轮 termination 分布：`'+json.dumps(dict(Counter(r['trajectory']['termination_reason'] for r in initial_rows)),ensure_ascii=False)+'`。', '']
    needed=max(0,3-sum(not r['success'] for r in final_rows))
    # If the final run has fewer than three failures, use actual archived failures,
    # explicitly distinguished from the final baseline rather than fabricating cases.
    if needed:
        lines += ['### 补充首轮失败案例', '', '最终重跑失败样本不足三条，以下使用独立首轮的真实失败补充诊断；不计入最终结果。', '']
        selected=[]
        for level in ('L1','L2','L3'):
            row=next((r for r in initial_rows if r['difficulty']==level and not r['success']),None)
            if row is not None:selected.append(row)
            if len(selected)==needed:break
        for r in selected:
            lines += [f'首轮 {r["task_id"]}：', '', '```json', json.dumps(dict(
                raw_outputs=r['trajectory']['raw_outputs'], tool_calls=r['trajectory']['tool_calls'],
                final_answer=r['trajectory']['final_answer'], termination_reason=r['trajectory']['termination_reason']),ensure_ascii=False,indent=2), '```', '']
    return '\n'.join(lines)


def reference_examples(rows):
    lines=['', '## 补充成功路径：环境参考脚本，非 Qwen baseline', '',
        '正式模型成功案例不足两条，不能补造真实模型成功案例。以下两条是已执行的可达性测试参考轨迹，分别覆盖 L2/L3；使用离线任务条件，不是自主策略，完全不计入模型成功率。', '']
    for level in ('L2','L3'):
        r=next(row for row in rows if row['difficulty']==level and row['success'])
        lines += [f'### 参考成功 {r["task_id"]}', '',
            f'tool_calls={r["tool_calls"]}，cost={r["tool_cost"]}，violations={r["rule_violation_count"]}。', '',
            '```json', json.dumps(dict(tool_calls=r['trajectory']['tool_calls'],final_answer=r['trajectory']['final_answer'],assertion_results=r['assertion_results']),ensure_ascii=False,indent=2), '```', '']
    return '\n'.join(lines)


def main():
    output=ROOT.parent/'results/env_v1_baseline.jsonl'
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    summary=json.loads(output.with_suffix('.summary.json').read_text())
    if len(rows)!=40 or summary['task_count']!=40:raise ValueError('Full 40-task baseline required')
    report=render(rows,summary)
    initial=output.with_name('env_v1_initial_protocol.jsonl')
    if initial.exists():
        initial_rows=[json.loads(line) for line in initial.read_text().splitlines()]
        initial_summary=json.loads(initial.with_suffix('.summary.json').read_text())
        report+=protocol_comparison(initial_rows,initial_summary,rows)
    reference=output.with_name('env_v1_reference.jsonl')
    if sum(r['success'] for r in rows)<2 and reference.exists():
        report+=reference_examples([json.loads(line) for line in reference.read_text().splitlines()])
    (ROOT.parent/'ENV_V1_RESULTS.md').write_text(report)


if __name__=='__main__':main()
