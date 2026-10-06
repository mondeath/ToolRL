# Environment v1 结果

## 改动与兼容

新增 `src/env_v1/`，保留原环境、旧任务、奖励和训练逻辑。复用 QueryEngine、Trajectory、public_task、HuggingFacePolicy 与 rollout；rollout 仅新增环境可选 schema/parser，消息构造按 schema 版本选择 prompt。

固定 KB seed=42：30 students、12 courses、90 enrollments、5 scholarships、5 初始 applications。课程完成与课程 ID 在同一 enrollment 上匹配。每个 episode 深拷贝并恢复初始状态。

七个 action：search / lookup / aggregate / verify 成本 1，submit 成本 2，deep_search 成本 5，direct_answer 成本 0。submit 支持显式 ID 批次并逐项拒绝无效/不合格/重复申请；一次批次成本 2，使完整 L3 可在 5 次调用内完成。deep_search 是合法的昂贵关系查询快捷路径，不强制惩罚工具选择。

旧项目无独立 verifier 和 failure injection，因此新增 eligibility 与最终状态检查，未替换已有奖励接口。search 使用完整的小表结果，无分页工具；L2 cheap path 经三次检索完成，deep_search 可以一次完成同一查询。此权衡是成本实验设计的一部分。

## 任务与评测

| 难度 | 数量 | 内容 | 参考工具调用数 |
|---|---:|---|---:|
| L1 | 10 | 5 GPA、4 year 列表、1 course count | 1 |
| L2 | 15 | 5 scholarship 资格、10 course/year/GPA 组合 | 3 |
| L3 | 15 | 5 scholarship × 3 独立已有申请状态 | 5 |

每个 L3 有 6 个本课程候选：3 名符合资格（1 名已经申请、2 名待提交），以及 GPA 或完成状态不满足的候选。L3 的三个变体共享题目文本但初始申请者不同；它们是状态变体，不是 15 个独立任务模板。

公开任务仍只有 task_id/question；任务文件的 evaluation 条件与 final_assertions 不进入 policy。L2 不保存最终答案，评测从初始 KB 计算；L3 显式比较完整 applications，并检查其他表和已有申请不可修改。参考路径只用于测试可达性，不参与 baseline 决策。

task_success 要求最终答案/状态正确、正常 direct_answer 结束且无非法状态变化；拒绝成功的重复/不合格尝试另记 violation，因此 task_success 与零违规不等价。trajectory_length 包含 direct_answer，tool_calls 不包含；parse repair 另存 parse_status。recovery_success 仅针对有注入事件的 episode，表示其最终成功率；无注入为 null。

## 真实 Qwen baseline

模型：Qwen2.5-1.5B-Instruct；revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。CPU float32，greedy decoding（do_sample=false），每轮最多 256 新 token，最大上下文 16384，最多 8 次工具调用/9 次决策，每次最多一次格式修复。failure_rate=0，40 题各一次，未筛选成功轨迹。

运行耗时 3093.6 秒。完整配置、运行版本、KB/task/prompt 哈希见 `results/env_v1_baseline.summary.json`。

| 指标 | 实测 |
|---|---:|
| overall success | 2.5% |
| L1 success | 10.0% |
| L2 success | 0.0% |
| L3 success | 0.0% |
| average tool cost | 30.175 |
| average trajectory length | 7.475 |
| average tool calls | 6.475 |
| deep_search usage（至少一次的 episode 比例） | 77.5% |
| deep_search calls | 237 |
| rule violation rate（有违规的 episode 比例） | 2.5% |
| rule violation count | 1 |
| recovery_success | null（本轮无注入） |

## 模型失败分析

termination 分布：`{"direct_answer": 1, "invalid_action": 8, "max_tool_calls": 31}`。

| 难度 | 平均成本 | 平均工具调用数 |
|---|---:|---:|
| L1 | 2.300 | 1.900 |
| L2 | 38.933 | 8.000 |
| L3 | 40.000 | 8.000 |

工具实际调用分布：`{"deep_search": 237, "lookup": 3, "search": 15, "verify": 4}`。预算耗尽与解析失败均按失败计入分母。低违规率也可能只是未执行 submit，不能单独解释成模型能够安全完成申请。

## 典型轨迹

实际仅有 1 条可展示的成功轨迹；不补造样本。

### 成功：v1-L1-00

What is the GPA of student s00? Return a number.

cost=6，tool_calls=2，violations=1，termination=direct_answer。

```json
{
  "tool_calls": [
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "s00"
        }
      },
      "observation": {
        "error": "invalid_entity",
        "entity": "s00"
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "lookup",
      "arguments": {
        "entity": "s00"
      },
      "observation": {
        "record": {
          "id": "s00",
          "name": "Student-00",
          "year": 1,
          "gpa": 3.8,
          "department": "Logic"
        }
      },
      "cost": 1,
      "failure_type": null
    }
  ],
  "final_answer": 3.8,
  "raw_outputs": [
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"s00\"}}}",
    "{\"action\":\"lookup\",\"arguments\":{\"entity\":\"s00\"}}",
    "{\"action\":\"direct_answer\",\"arguments\":{\"final_answer\":3.8}}"
  ],
  "rule_violations": [
    {
      "type": "invalid_entity",
      "entity": "s00"
    }
  ]
}
```

### 失败：v1-L1-01

What is the GPA of student s01? Return a number.

cost=1，tool_calls=1，violations=0，termination=invalid_action。

```json
{
  "tool_calls": [
    {
      "tool_name": "lookup",
      "arguments": {
        "entity": "s01"
      },
      "observation": {
        "record": {
          "id": "s01",
          "name": "Student-01",
          "year": 1,
          "gpa": 3.7,
          "department": "Logic"
        }
      },
      "cost": 1,
      "failure_type": null
    }
  ],
  "final_answer": null,
  "raw_outputs": [
    "{\"action\":\"lookup\",\"arguments\":{\"entity\":\"s01\"}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"none\",\"year\":null,\"min_gpa\":null,\"course_id\":\"none\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"none\",\"year\":null,\"min_gpa\":null,\"course_id\":\"none\"}}}"
  ],
  "rule_violations": []
}
```

### 失败：v1-L2-00

List all student IDs eligible for scholarship sc00, including those already applied. Check year, GPA and completed required course.

cost=36，tool_calls=8，violations=0，termination=max_tool_calls。

```json
{
  "tool_calls": [
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "verify",
      "arguments": {
        "student_ids": [
          "s00",
          "s01",
          "s02"
        ],
        "scholarship_id": "sc00"
      },
      "observation": {
        "results": [
          {
            "student_id": "s00",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": true
          },
          {
            "student_id": "s01",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": false
          },
          {
            "student_id": "s02",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": false
          }
        ]
      },
      "cost": 1,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    }
  ],
  "final_answer": null,
  "raw_outputs": [
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"verify\",\"arguments\":{\"student_ids\":[\"s00\",\"s01\",\"s02\"],\"scholarship_id\":\"sc00\"}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}"
  ],
  "rule_violations": []
}
```

### 失败：v1-L3-00

Submit scholarship sc00 applications for every eligible student who has not already applied. Do not submit duplicates or ineligible students. Preserve all existing records. Then finish with direct_answer.

cost=40，tool_calls=8，violations=0，termination=max_tool_calls。

```json
{
  "tool_calls": [
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "scholarship_id": "sc00"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02"
        ]
      },
      "cost": 5,
      "failure_type": null
    }
  ],
  "final_answer": null,
  "raw_outputs": [
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"scholarship_id\":\"sc00\"}}}"
  ],
  "rule_violations": []
}
```

## 验证与失败注入

pytest 的实际执行记录见 `results/env_v1_tests.txt`，原测试与新增测试全部通过。40 条测试专用参考路径（`results/env_v1_reference.jsonl`，明确标注非模型结果）全部成功且无违规，L2/L3 为 3–5 tool calls；这证明环境可解，不是模型性能。新增测试覆盖搜索/完整 lookup/聚合/关系查询/eligibility/重复与不合格提交/最终状态/非法修改/成本/reset 隔离/解析/rollout/timeout/partial_result/恢复。

旧 smoke 在临时目录跑完 192 条轨迹，economical / redundant_deep 均为 100%，未覆盖原有数据；结果见 `results/env_v1_legacy_smoke.json`。

timeout 执行前返回失败，不产生写入；partial_result 执行后裁剪结果并显式标记，写入可能已经成功。测试验证 partial submit 后先 verify 再决定重试。

`results/env_v1_failure_smoke.json` 为固定 seed 的脚本 lookup 重试探针，分别以 0/0.2/0.5 概率跑 50 次；它不衡量 Qwen 的恢复能力。真实模型本轮只跑 failure_rate=0，后续可用同一 runner 的 --failure-rate 复跑。

## 是否适合进入 GRPO

至少一个 L2/L3 成功率低于 20%，尚不满足直接进入 GRPO 的难度目标。需先分析协议错误、候选筛选和状态操作失败，冻结明确的标量/列表输出示例后另开实验验证，保留本轮结果。

当前仅 40 条、单 seed、greedy baseline，且 L3 同奖学金任务共享结构；不能据此推断泛化性能。先固定 prompt 与评测规则，补真实模型 0.2/0.5 failure 实验及小规模独立种子验证，再讨论 reward 权重与 GRPO。没有实现 GRPO、引入新模型、user simulator 或大型框架。

## 复现

```bash
.venv/bin/python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiment/scripts/run_env_v1.py
.venv/bin/python experiment/scripts/check_env_v1_failures.py
.venv/bin/python experiment/scripts/report_env_v1.py
```

## 首轮协议诊断（独立保存，不混入最终指标）

首轮同样跑完 40 题，overall success=0.0%，L1/L2/L3=0.0%/0.0%/0.0%。

首轮使用简写参数示例，模型常把 deep_search 条件放到 arguments 顶层，漏掉 query；还出现标量返回单元素数组、反复执行同一查询。首轮未修改或删去，见 `results/env_v1_initial_protocol.jsonl` 与配套 summary/KB/tasks/log；提示词原文见 `results/env_v1_initial_protocol_prompt.txt`，已与首轮 prompt 哈希核对。

正式重跑仅更新 v1 system prompt：改为每种 action 的完整 JSON 示例，明确 query 嵌套、标量/列表格式、不得混合两种关系查询格式，以及完成检索后结束而不重复确定性查询。没有改变任务、数据库、工具、评分规则、模型或生成参数。这是基于同一任务集做的协议调试，不能视作独立留出评估，也不是 RL 改进。最终 prompt/messages 保存于 `results/env_v1_baseline.prompt.json`。

首轮 termination 分布：`{"direct_answer": 5, "invalid_action": 32, "max_tool_calls": 3}`。

## 补充成功路径：环境参考脚本，非 Qwen baseline

正式模型成功案例不足两条，不能补造真实模型成功案例。以下两条是已执行的可达性测试参考轨迹，分别覆盖 L2/L3；使用离线任务条件，不是自主策略，完全不计入模型成功率。

### 参考成功 v1-L2-00

tool_calls=3，cost=3，violations=0。

```json
{
  "tool_calls": [
    {
      "tool_name": "lookup",
      "arguments": {
        "entity": "sc00"
      },
      "observation": {
        "record": {
          "id": "sc00",
          "name": "Scholarship-00",
          "min_gpa": 3.5,
          "eligible_year": 1,
          "required_course": "c00"
        }
      },
      "cost": 1,
      "failure_type": null
    },
    {
      "tool_name": "search",
      "arguments": {
        "query": {
          "table": "enrollments",
          "where": [
            {
              "field": "course_id",
              "value": "c00"
            }
          ],
          "select": "student_id"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02",
          "s03",
          "s04",
          "s05"
        ],
        "results": [
          {
            "entity": "e00-0",
            "values": [
              "s00"
            ]
          },
          {
            "entity": "e01-0",
            "values": [
              "s01"
            ]
          },
          {
            "entity": "e02-0",
            "values": [
              "s02"
            ]
          },
          {
            "entity": "e03-0",
            "values": [
              "s03"
            ]
          },
          {
            "entity": "e04-0",
            "values": [
              "s04"
            ]
          },
          {
            "entity": "e05-0",
            "values": [
              "s05"
            ]
          }
        ],
        "truncated": false,
        "matched_count": 6
      },
      "cost": 1,
      "failure_type": null
    },
    {
      "tool_name": "verify",
      "arguments": {
        "student_ids": [
          "s00",
          "s01",
          "s02",
          "s03",
          "s04",
          "s05"
        ],
        "scholarship_id": "sc00"
      },
      "observation": {
        "results": [
          {
            "student_id": "s00",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": true
          },
          {
            "student_id": "s01",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": false
          },
          {
            "student_id": "s02",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": false
          },
          {
            "student_id": "s03",
            "scholarship_id": "sc00",
            "eligible": false,
            "checks": {
              "gpa": false,
              "year": true,
              "course": true
            },
            "already_applied": false
          },
          {
            "student_id": "s04",
            "scholarship_id": "sc00",
            "eligible": false,
            "checks": {
              "gpa": true,
              "year": true,
              "course": false
            },
            "already_applied": false
          },
          {
            "student_id": "s05",
            "scholarship_id": "sc00",
            "eligible": false,
            "checks": {
              "gpa": false,
              "year": true,
              "course": true
            },
            "already_applied": false
          }
        ]
      },
      "cost": 1,
      "failure_type": null
    }
  ],
  "final_answer": [
    "s00",
    "s01",
    "s02"
  ],
  "assertion_results": []
}
```

### 参考成功 v1-L3-00

tool_calls=5，cost=6，violations=0。

```json
{
  "tool_calls": [
    {
      "tool_name": "lookup",
      "arguments": {
        "entity": "sc00"
      },
      "observation": {
        "record": {
          "id": "sc00",
          "name": "Scholarship-00",
          "min_gpa": 3.5,
          "eligible_year": 1,
          "required_course": "c00"
        }
      },
      "cost": 1,
      "failure_type": null
    },
    {
      "tool_name": "search",
      "arguments": {
        "query": {
          "table": "enrollments",
          "where": [
            {
              "field": "course_id",
              "value": "c00"
            }
          ],
          "select": "student_id"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02",
          "s03",
          "s04",
          "s05"
        ],
        "results": [
          {
            "entity": "e00-0",
            "values": [
              "s00"
            ]
          },
          {
            "entity": "e01-0",
            "values": [
              "s01"
            ]
          },
          {
            "entity": "e02-0",
            "values": [
              "s02"
            ]
          },
          {
            "entity": "e03-0",
            "values": [
              "s03"
            ]
          },
          {
            "entity": "e04-0",
            "values": [
              "s04"
            ]
          },
          {
            "entity": "e05-0",
            "values": [
              "s05"
            ]
          }
        ],
        "truncated": false,
        "matched_count": 6
      },
      "cost": 1,
      "failure_type": null
    },
    {
      "tool_name": "verify",
      "arguments": {
        "student_ids": [
          "s00",
          "s01",
          "s02",
          "s03",
          "s04",
          "s05"
        ],
        "scholarship_id": "sc00"
      },
      "observation": {
        "results": [
          {
            "student_id": "s00",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": true
          },
          {
            "student_id": "s01",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": false
          },
          {
            "student_id": "s02",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": false
          },
          {
            "student_id": "s03",
            "scholarship_id": "sc00",
            "eligible": false,
            "checks": {
              "gpa": false,
              "year": true,
              "course": true
            },
            "already_applied": false
          },
          {
            "student_id": "s04",
            "scholarship_id": "sc00",
            "eligible": false,
            "checks": {
              "gpa": true,
              "year": true,
              "course": false
            },
            "already_applied": false
          },
          {
            "student_id": "s05",
            "scholarship_id": "sc00",
            "eligible": false,
            "checks": {
              "gpa": false,
              "year": true,
              "course": true
            },
            "already_applied": false
          }
        ]
      },
      "cost": 1,
      "failure_type": null
    },
    {
      "tool_name": "submit",
      "arguments": {
        "student_ids": [
          "s01",
          "s02"
        ],
        "scholarship_id": "sc00"
      },
      "observation": {
        "results": [
          {
            "student_id": "s01",
            "scholarship_id": "sc00",
            "status": "submitted"
          },
          {
            "student_id": "s02",
            "scholarship_id": "sc00",
            "status": "submitted"
          }
        ]
      },
      "cost": 2,
      "failure_type": null
    },
    {
      "tool_name": "verify",
      "arguments": {
        "student_ids": [
          "s00",
          "s01",
          "s02"
        ],
        "scholarship_id": "sc00"
      },
      "observation": {
        "results": [
          {
            "student_id": "s00",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": true
          },
          {
            "student_id": "s01",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": true
          },
          {
            "student_id": "s02",
            "scholarship_id": "sc00",
            "eligible": true,
            "checks": {
              "gpa": true,
              "year": true,
              "course": true
            },
            "already_applied": true
          }
        ]
      },
      "cost": 1,
      "failure_type": null
    }
  ],
  "final_answer": [
    "s00",
    "s01",
    "s02"
  ],
  "assertion_results": [
    true,
    true,
    true
  ]
}
```
