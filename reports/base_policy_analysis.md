# Base Policy Tool-Use Distribution

真实 HuggingFace 推理诊断，无 AI Judge、reward 优化或训练。CPU smoke 只检验接口，不代表 40–80 条正式实验。

## Run configuration

```json
{
  "config": {
    "policy": {
      "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
      "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "device": "cpu",
      "dtype": "float32",
      "cpu_fallback": true,
      "temperature": 0.8,
      "top_p": 0.95,
      "max_new_tokens": 192,
      "max_context_tokens": 8192,
      "do_sample": true,
      "cache_dir": "../.hf-cache/hub",
      "local_files_only": true,
      "cpu_threads": 4
    },
    "rollout": {
      "max_steps": 4,
      "max_tool_calls": 3,
      "max_parse_retries": 1,
      "seed": 2026
    },
    "environment": {
      "top_k": 5
    },
    "task_count": 4,
    "group_size": 2,
    "output_dir": "data/cpu_smoke",
    "report_path": "reports/base_policy_analysis_cpu_smoke.md"
  },
  "runtime": {
    "device": "cpu",
    "dtype": "float32",
    "torch_version": "2.8.0",
    "transformers_version": "4.57.6",
    "resolved_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
  },
  "elapsed_seconds": 367.22858214378357,
  "task_ids": [
    "task-032",
    "task-033",
    "task-000",
    "task-001"
  ],
  "kb_sha256": "fa01f5ceded2f75eb053df2045c147d48c0376dc1aee4ef22e44512bf3283775",
  "public_tasks_sha256": "83cf8ea10d1b3643805a14443c2ec7566185e86674d718c7e65194cae1064c23",
  "evaluation_tasks_sha256": "158385f6cf1afd615d65d0dca9f4656cf17e8210559f8ab3be7a340b00b139a9",
  "prompt_sha256": "cc4b2d9210690e8181434fb19c0f366432036cb07fa6b947ed9fc0fe9b1bb9bf"
}
```

## Aggregate

```json
{
  "num_trajectories": 8,
  "normalized_accuracy": 0.125,
  "average_cost": 12.875,
  "average_tool_calls": 2.125,
  "deep_search_usage_rate": 1.0,
  "direct_answer_without_tool_rate": 0.0,
  "invalid_action_rate": 0.1111111111111111,
  "episodes_with_invalid_action_rate": 0.125,
  "truncation_rate": 0.125,
  "tool_execution_errors": 3,
  "termination_counts": {
    "direct_answer": 6,
    "max_tool_calls": 1,
    "invalid_action": 1
  }
}
```

## Task type

| Type | N | Accuracy | Average cost |
|---|---:|---:|---:|
| list/filter | 2 | 0.000 | 17.000 |
| multi-condition | 2 | 0.000 | 17.000 |
| single-hop | 2 | 0.000 | 8.500 |
| two-hop | 2 | 0.500 | 9.000 |

## Tool / action usage

action_usage 包括终止 action；tool_call_usage 仅以已执行工具调用为分母。

proposed_action_counts 包含可解析出 action 名称但参数验证失败的尝试；unparsed 表示不能可靠提取名称。不要将零次实际调用误解为没有使用工具的意图。

```json
{
  "action_usage": {
    "lookup": {
      "count": 3,
      "fraction": 0.13043478260869565
    },
    "search": {
      "count": 2,
      "fraction": 0.08695652173913043
    },
    "deep_search": {
      "count": 12,
      "fraction": 0.5217391304347826
    },
    "direct_answer": {
      "count": 6,
      "fraction": 0.2608695652173913
    }
  },
  "tool_call_usage": {
    "lookup": {
      "count": 3,
      "fraction": 0.17647058823529413
    },
    "search": {
      "count": 2,
      "fraction": 0.11764705882352941
    },
    "deep_search": {
      "count": 12,
      "fraction": 0.7058823529411765
    }
  },
  "first_action_counts": {
    "deep_search": 7,
    "search": 1
  },
  "proposed_action_counts": {
    "deep_search": 15,
    "direct_answer": 6,
    "search": 3,
    "lookup": 3
  },
  "first_proposed_action_counts": {
    "deep_search": 7,
    "search": 1
  }
}
```

## Group differences

```json
{
  "task-032": {
    "group_size": 2,
    "unique_tool_sequences": 2,
    "unique_final_answers": 1,
    "missing_final_answers": 1,
    "cost_variance": 1,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [
          "deep_search",
          "deep_search",
          "direct_answer"
        ],
        "final_answer": "[\"s01\", \"s02\", \"s03\", \"s07\", \"s13\", \"s17\", \"s20\", \"s21\", \"s25\", \"s28\"]",
        "cost": 16,
        "termination_reason": "direct_answer"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [
          "deep_search",
          "deep_search",
          "search"
        ],
        "final_answer": null,
        "cost": 18,
        "termination_reason": "max_tool_calls"
      }
    ]
  },
  "task-033": {
    "group_size": 2,
    "unique_tool_sequences": 2,
    "unique_final_answers": 2,
    "missing_final_answers": 0,
    "cost_variance": 1,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [
          "search",
          "deep_search",
          "deep_search",
          "direct_answer"
        ],
        "final_answer": "[\"student_id_1\", \"student_id_2\"]",
        "cost": 18,
        "termination_reason": "direct_answer"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [
          "deep_search",
          "deep_search",
          "direct_answer"
        ],
        "final_answer": "[\"s00\", \"s01\", \"s02\", \"s03\", \"s04\", \"s05\", \"s06\", \"s07\", \"s08\", \"s09\", \"s10\", \"s11\", \"s12\", \"s13\", \"s14\", \"s15\", \"s16\", \"s17\", \"s18\", \"s19\", \"s20\", \"s21\", \"s22\", \"s23\", \"s24\", \"s25\", \"s26\", \"s27\", \"s28\", \"s29\"]",
        "cost": 16,
        "termination_reason": "direct_answer"
      }
    ]
  },
  "task-000": {
    "group_size": 2,
    "unique_tool_sequences": 2,
    "unique_final_answers": 2,
    "missing_final_answers": 0,
    "cost_variance": 0.25,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [
          "deep_search",
          "lookup",
          "direct_answer"
        ],
        "final_answer": "[]",
        "cost": 9,
        "termination_reason": "direct_answer"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [
          "deep_search",
          "direct_answer"
        ],
        "final_answer": "[\"s00\"]",
        "cost": 8,
        "termination_reason": "direct_answer"
      }
    ]
  },
  "task-001": {
    "group_size": 2,
    "unique_tool_sequences": 2,
    "unique_final_answers": 1,
    "missing_final_answers": 1,
    "cost_variance": 0,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [
          "deep_search",
          "lookup"
        ],
        "final_answer": null,
        "cost": 9,
        "termination_reason": "invalid_action"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [
          "deep_search",
          "lookup",
          "direct_answer"
        ],
        "final_answer": "Ecology",
        "cost": 9,
        "termination_reason": "direct_answer"
      }
    ]
  }
}
```

## Invalid / truncated

invalid_action_rate = 解析或 schema 验证失败次数 / 所有模型 action 输出次数（包括 repair）。达到最终解析失败时以 invalid_action 结束，不自动编造答案；truncated 专指预算/上下文/推理中断。工具执行错误另外计数。

```json
[
  {
    "task_id": "task-032",
    "task_type": "list/filter",
    "rollout_id": 1,
    "correctness": false,
    "total_cost": 18,
    "number_of_tool_calls": 3,
    "deep_search_used": true,
    "direct_answer_without_tool": false,
    "termination_reason": "max_tool_calls",
    "truncated": true,
    "invalid_action_attempts": 0,
    "action_attempts": 4,
    "invalid_arguments": 0,
    "repair_attempts": 0,
    "tool_execution_errors": 0,
    "tool_sequence": [
      "deep_search",
      "deep_search",
      "search"
    ],
    "final_answer": null
  },
  {
    "task_id": "task-033",
    "task_type": "multi-condition",
    "rollout_id": 0,
    "correctness": false,
    "total_cost": 18,
    "number_of_tool_calls": 3,
    "deep_search_used": true,
    "direct_answer_without_tool": false,
    "termination_reason": "direct_answer",
    "truncated": false,
    "invalid_action_attempts": 0,
    "action_attempts": 4,
    "invalid_arguments": 0,
    "repair_attempts": 0,
    "tool_execution_errors": 3,
    "tool_sequence": [
      "search",
      "deep_search",
      "deep_search",
      "direct_answer"
    ],
    "final_answer": "[\"student_id_1\", \"student_id_2\"]"
  },
  {
    "task_id": "task-001",
    "task_type": "two-hop",
    "rollout_id": 0,
    "correctness": false,
    "total_cost": 9,
    "number_of_tool_calls": 2,
    "deep_search_used": true,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 3,
    "action_attempts": 5,
    "invalid_arguments": 3,
    "repair_attempts": 2,
    "tool_execution_errors": 0,
    "tool_sequence": [
      "deep_search",
      "lookup"
    ],
    "final_answer": null
  }
]
```

## Observed interface signals

以下仅标记重复的相同调用和列表样式字符串，不自动修正模型答案或影响 policy。

```json
{
  "episodes_with_identical_repeated_calls": [
    {
      "task_id": "task-032",
      "rollout_id": 0,
      "repeated_calls": 1
    },
    {
      "task_id": "task-033",
      "rollout_id": 0,
      "repeated_calls": 1
    }
  ],
  "list_like_string_answers": [
    {
      "task_id": "task-032",
      "rollout_id": 0
    },
    {
      "task_id": "task-033",
      "rollout_id": 0
    },
    {
      "task_id": "task-033",
      "rollout_id": 1
    },
    {
      "task_id": "task-000",
      "rollout_id": 0
    },
    {
      "task_id": "task-000",
      "rollout_id": 1
    }
  ],
  "parse_error_counts": {
    "Invalid operator": 1,
    "Unknown field for table: advisor_id": 1,
    "Unknown table": 1
  }
}
```

## Interface limitations

- search 的 top-k 无分页；列表题可能需要 deep_search。工具能力不对称会影响初始偏好。

- deep_search 返回数据库查询结果和较多扫描证据；长 observation 可能耗尽上下文。当前显式终止，不静默裁掉历史。

- keyword search 是子串查询；自然语言问题不能直接当作 deep_search query。

- schema 合法不等于存在对应实体；此类错误由工具报告，并按原环境规则计费。

- 最终答案只接受标量或标量数组；正确率不理解自然语言解释。

- 独立随机种子不保证行为不同，低 diversity 本身是诊断结果。

## Five complete trajectory examples

### Example 1: task-032 / rollout 0

```json
{
  "task_id": "task-032",
  "question": "List student IDs in year 1.",
  "steps": [
    {
      "action": "deep_search",
      "tool_call_index": 0,
      "cost": 8
    },
    {
      "action": "deep_search",
      "tool_call_index": 1,
      "cost": 8
    },
    {
      "action": "direct_answer",
      "arguments": {
        "final_answer": "[\"s01\", \"s02\", \"s03\", \"s07\", \"s13\", \"s17\", \"s20\", \"s21\", \"s25\", \"s28\"]"
      },
      "cost": 0
    }
  ],
  "tool_calls": [
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "year",
              "op": "eq",
              "value": 1
            }
          ],
          "select": "id"
        }
      },
      "observation": {
        "answer": [
          "s01",
          "s02",
          "s03",
          "s07",
          "s13",
          "s17",
          "s20",
          "s21",
          "s25",
          "s28"
        ],
        "results": [
          {
            "entity": "s01",
            "values": [
              "s01"
            ]
          },
          {
            "entity": "s02",
            "values": [
              "s02"
            ]
          },
          {
            "entity": "s03",
            "values": [
              "s03"
            ]
          },
          {
            "entity": "s07",
            "values": [
              "s07"
            ]
          },
          {
            "entity": "s13",
            "values": [
              "s13"
            ]
          },
          {
            "entity": "s17",
            "values": [
              "s17"
            ]
          },
          {
            "entity": "s20",
            "values": [
              "s20"
            ]
          },
          {
            "entity": "s21",
            "values": [
              "s21"
            ]
          },
          {
            "entity": "s25",
            "values": [
              "s25"
            ]
          },
          {
            "entity": "s28",
            "values": [
              "s28"
            ]
          }
        ],
        "evidence": [
          {
            "entity": "s00",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s01",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s01",
            "field": "id",
            "value": "s01"
          },
          {
            "entity": "s02",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s02",
            "field": "id",
            "value": "s02"
          },
          {
            "entity": "s03",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s03",
            "field": "id",
            "value": "s03"
          },
          {
            "entity": "s04",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s05",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s06",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s07",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s07",
            "field": "id",
            "value": "s07"
          },
          {
            "entity": "s08",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s09",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s10",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s11",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s12",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s13",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s13",
            "field": "id",
            "value": "s13"
          },
          {
            "entity": "s14",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s15",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s16",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s17",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s17",
            "field": "id",
            "value": "s17"
          },
          {
            "entity": "s18",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s19",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s20",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s20",
            "field": "id",
            "value": "s20"
          },
          {
            "entity": "s21",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s21",
            "field": "id",
            "value": "s21"
          },
          {
            "entity": "s22",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s23",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s24",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s25",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s25",
            "field": "id",
            "value": "s25"
          },
          {
            "entity": "s26",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s27",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s28",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s28",
            "field": "id",
            "value": "s28"
          },
          {
            "entity": "s29",
            "field": "year",
            "value": 4
          }
        ],
        "truncated": false,
        "matched_count": 10
      },
      "cost": 8
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "year",
              "op": "eq",
              "value": 1
            }
          ],
          "select": "id"
        }
      },
      "observation": {
        "answer": [
          "s01",
          "s02",
          "s03",
          "s07",
          "s13",
          "s17",
          "s20",
          "s21",
          "s25",
          "s28"
        ],
        "results": [
          {
            "entity": "s01",
            "values": [
              "s01"
            ]
          },
          {
            "entity": "s02",
            "values": [
              "s02"
            ]
          },
          {
            "entity": "s03",
            "values": [
              "s03"
            ]
          },
          {
            "entity": "s07",
            "values": [
              "s07"
            ]
          },
          {
            "entity": "s13",
            "values": [
              "s13"
            ]
          },
          {
            "entity": "s17",
            "values": [
              "s17"
            ]
          },
          {
            "entity": "s20",
            "values": [
              "s20"
            ]
          },
          {
            "entity": "s21",
            "values": [
              "s21"
            ]
          },
          {
            "entity": "s25",
            "values": [
              "s25"
            ]
          },
          {
            "entity": "s28",
            "values": [
              "s28"
            ]
          }
        ],
        "evidence": [
          {
            "entity": "s00",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s01",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s01",
            "field": "id",
            "value": "s01"
          },
          {
            "entity": "s02",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s02",
            "field": "id",
            "value": "s02"
          },
          {
            "entity": "s03",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s03",
            "field": "id",
            "value": "s03"
          },
          {
            "entity": "s04",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s05",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s06",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s07",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s07",
            "field": "id",
            "value": "s07"
          },
          {
            "entity": "s08",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s09",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s10",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s11",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s12",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s13",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s13",
            "field": "id",
            "value": "s13"
          },
          {
            "entity": "s14",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s15",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s16",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s17",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s17",
            "field": "id",
            "value": "s17"
          },
          {
            "entity": "s18",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s19",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s20",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s20",
            "field": "id",
            "value": "s20"
          },
          {
            "entity": "s21",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s21",
            "field": "id",
            "value": "s21"
          },
          {
            "entity": "s22",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s23",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s24",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s25",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s25",
            "field": "id",
            "value": "s25"
          },
          {
            "entity": "s26",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s27",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s28",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s28",
            "field": "id",
            "value": "s28"
          },
          {
            "entity": "s29",
            "field": "year",
            "value": 4
          }
        ],
        "truncated": false,
        "matched_count": 10
      },
      "cost": 8
    }
  ],
  "observations": [
    {
      "answer": [
        "s01",
        "s02",
        "s03",
        "s07",
        "s13",
        "s17",
        "s20",
        "s21",
        "s25",
        "s28"
      ],
      "results": [
        {
          "entity": "s01",
          "values": [
            "s01"
          ]
        },
        {
          "entity": "s02",
          "values": [
            "s02"
          ]
        },
        {
          "entity": "s03",
          "values": [
            "s03"
          ]
        },
        {
          "entity": "s07",
          "values": [
            "s07"
          ]
        },
        {
          "entity": "s13",
          "values": [
            "s13"
          ]
        },
        {
          "entity": "s17",
          "values": [
            "s17"
          ]
        },
        {
          "entity": "s20",
          "values": [
            "s20"
          ]
        },
        {
          "entity": "s21",
          "values": [
            "s21"
          ]
        },
        {
          "entity": "s25",
          "values": [
            "s25"
          ]
        },
        {
          "entity": "s28",
          "values": [
            "s28"
          ]
        }
      ],
      "evidence": [
        {
          "entity": "s00",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s01",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s01",
          "field": "id",
          "value": "s01"
        },
        {
          "entity": "s02",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s02",
          "field": "id",
          "value": "s02"
        },
        {
          "entity": "s03",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s03",
          "field": "id",
          "value": "s03"
        },
        {
          "entity": "s04",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s05",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s06",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s07",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s07",
          "field": "id",
          "value": "s07"
        },
        {
          "entity": "s08",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s09",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s10",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s11",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s12",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s13",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s13",
          "field": "id",
          "value": "s13"
        },
        {
          "entity": "s14",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s15",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s16",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s17",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s17",
          "field": "id",
          "value": "s17"
        },
        {
          "entity": "s18",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s19",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s20",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s20",
          "field": "id",
          "value": "s20"
        },
        {
          "entity": "s21",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s21",
          "field": "id",
          "value": "s21"
        },
        {
          "entity": "s22",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s23",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s24",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s25",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s25",
          "field": "id",
          "value": "s25"
        },
        {
          "entity": "s26",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s27",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s28",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s28",
          "field": "id",
          "value": "s28"
        },
        {
          "entity": "s29",
          "field": "year",
          "value": 4
        }
      ],
      "truncated": false,
      "matched_count": 10
    },
    {
      "answer": [
        "s01",
        "s02",
        "s03",
        "s07",
        "s13",
        "s17",
        "s20",
        "s21",
        "s25",
        "s28"
      ],
      "results": [
        {
          "entity": "s01",
          "values": [
            "s01"
          ]
        },
        {
          "entity": "s02",
          "values": [
            "s02"
          ]
        },
        {
          "entity": "s03",
          "values": [
            "s03"
          ]
        },
        {
          "entity": "s07",
          "values": [
            "s07"
          ]
        },
        {
          "entity": "s13",
          "values": [
            "s13"
          ]
        },
        {
          "entity": "s17",
          "values": [
            "s17"
          ]
        },
        {
          "entity": "s20",
          "values": [
            "s20"
          ]
        },
        {
          "entity": "s21",
          "values": [
            "s21"
          ]
        },
        {
          "entity": "s25",
          "values": [
            "s25"
          ]
        },
        {
          "entity": "s28",
          "values": [
            "s28"
          ]
        }
      ],
      "evidence": [
        {
          "entity": "s00",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s01",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s01",
          "field": "id",
          "value": "s01"
        },
        {
          "entity": "s02",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s02",
          "field": "id",
          "value": "s02"
        },
        {
          "entity": "s03",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s03",
          "field": "id",
          "value": "s03"
        },
        {
          "entity": "s04",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s05",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s06",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s07",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s07",
          "field": "id",
          "value": "s07"
        },
        {
          "entity": "s08",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s09",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s10",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s11",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s12",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s13",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s13",
          "field": "id",
          "value": "s13"
        },
        {
          "entity": "s14",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s15",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s16",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s17",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s17",
          "field": "id",
          "value": "s17"
        },
        {
          "entity": "s18",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s19",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s20",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s20",
          "field": "id",
          "value": "s20"
        },
        {
          "entity": "s21",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s21",
          "field": "id",
          "value": "s21"
        },
        {
          "entity": "s22",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s23",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s24",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s25",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s25",
          "field": "id",
          "value": "s25"
        },
        {
          "entity": "s26",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s27",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s28",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s28",
          "field": "id",
          "value": "s28"
        },
        {
          "entity": "s29",
          "field": "year",
          "value": 4
        }
      ],
      "truncated": false,
      "matched_count": 10
    }
  ],
  "final_answer": "[\"s01\", \"s02\", \"s03\", \"s07\", \"s13\", \"s17\", \"s20\", \"s21\", \"s25\", \"s28\"]",
  "total_cost": 16,
  "metadata": {
    "rollout_id": 0,
    "rollout_config": {
      "max_steps": 4,
      "max_tool_calls": 3,
      "max_parse_retries": 1,
      "seed": 2026
    },
    "runtime": {
      "device": "cpu",
      "dtype": "float32",
      "torch_version": "2.8.0",
      "transformers_version": "4.57.6",
      "resolved_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    }
  },
  "policy_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "sampling_config": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 192,
    "do_sample": true,
    "seed": 1880828236724239200
  },
  "raw_outputs": [
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"year\", \"op\": \"eq\", \"value\": 1}], \"select\": \"id\"}}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"year\", \"op\": \"eq\", \"value\": 1}], \"select\": \"id\"}}}",
    "{\"action\": \"direct_answer\", \"arguments\": {\"final_answer\": \"[\\\"s01\\\", \\\"s02\\\", \\\"s03\\\", \\\"s07\\\", \\\"s13\\\", \\\"s17\\\", \\\"s20\\\", \\\"s21\\\", \\\"s25\\\", \\\"s28\\\"]\"}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "eq",
                "value": 1
              }
            ],
            "select": "id"
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 2,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "eq",
                "value": 1
              }
            ],
            "select": "id"
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 3,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "direct_answer",
        "arguments": {
          "final_answer": "[\"s01\", \"s02\", \"s03\", \"s07\", \"s13\", \"s17\", \"s20\", \"s21\", \"s25\", \"s28\"]"
        }
      },
      "error": null,
      "invalid_arguments": false
    }
  ],
  "termination_reason": "direct_answer",
  "truncated": false,
  "number_of_steps": 3,
  "tool_sequence": [
    "deep_search",
    "deep_search",
    "direct_answer"
  ],
  "token_ids": [],
  "action_token_mask": [],
  "logprobs": []
}
```

### Example 2: task-033 / rollout 0

```json
{
  "task_id": "task-033",
  "question": "List student IDs in year 1 with advisor p00.",
  "steps": [
    {
      "action": "search",
      "tool_call_index": 0,
      "cost": 2
    },
    {
      "action": "deep_search",
      "tool_call_index": 1,
      "cost": 8
    },
    {
      "action": "deep_search",
      "tool_call_index": 2,
      "cost": 8
    },
    {
      "action": "direct_answer",
      "arguments": {
        "final_answer": "[\"student_id_1\", \"student_id_2\"]"
      },
      "cost": 0
    }
  ],
  "tool_calls": [
    {
      "tool_name": "search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "year",
              "op": "gte",
              "value": "1"
            },
            {
              "field": "advisor",
              "op": "eq",
              "value": "p00"
            }
          ]
        }
      },
      "observation": {
        "error": "'>=' not supported between instances of 'int' and 'str'",
        "error_type": "TypeError"
      },
      "cost": 2
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "advisor",
              "op": "eq",
              "value": "p00"
            },
            {
              "field": "year",
              "op": "gte",
              "value": "1"
            }
          ],
          "select": "id",
          "reduce": "list"
        }
      },
      "observation": {
        "error": "'>=' not supported between instances of 'int' and 'str'",
        "error_type": "TypeError"
      },
      "cost": 8
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "advisor",
              "op": "eq",
              "value": "p00"
            },
            {
              "field": "year",
              "op": "gte",
              "value": "1"
            }
          ],
          "select": "id",
          "reduce": "list"
        }
      },
      "observation": {
        "error": "'>=' not supported between instances of 'int' and 'str'",
        "error_type": "TypeError"
      },
      "cost": 8
    }
  ],
  "observations": [
    {
      "error": "'>=' not supported between instances of 'int' and 'str'",
      "error_type": "TypeError"
    },
    {
      "error": "'>=' not supported between instances of 'int' and 'str'",
      "error_type": "TypeError"
    },
    {
      "error": "'>=' not supported between instances of 'int' and 'str'",
      "error_type": "TypeError"
    }
  ],
  "final_answer": "[\"student_id_1\", \"student_id_2\"]",
  "total_cost": 18,
  "metadata": {
    "rollout_id": 0,
    "rollout_config": {
      "max_steps": 4,
      "max_tool_calls": 3,
      "max_parse_retries": 1,
      "seed": 2026
    },
    "runtime": {
      "device": "cpu",
      "dtype": "float32",
      "torch_version": "2.8.0",
      "transformers_version": "4.57.6",
      "resolved_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    }
  },
  "policy_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "sampling_config": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 192,
    "do_sample": true,
    "seed": 2867593952370870543
  },
  "raw_outputs": [
    "{\"action\":\"search\",\"arguments\":{\"query\":{\"table\":\"students\",\"where\":[{\"field\":\"year\",\"op\":\"gte\",\"value\":\"1\"},{\"field\":\"advisor\",\"op\":\"eq\",\"value\":\"p00\"}]}}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"advisor\", \"op\": \"eq\", \"value\": \"p00\"}, {\"field\": \"year\", \"op\": \"gte\", \"value\": \"1\"}], \"select\": \"id\", \"reduce\": \"list\"}}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"advisor\", \"op\": \"eq\", \"value\": \"p00\"}, {\"field\": \"year\", \"op\": \"gte\", \"value\": \"1\"}], \"select\": \"id\", \"reduce\": \"list\"}}}",
    "{\"action\": \"direct_answer\", \"arguments\": {\"final_answer\": \"[\\\"student_id_1\\\", \\\"student_id_2\\\"]\"}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "gte",
                "value": "1"
              },
              {
                "field": "advisor",
                "op": "eq",
                "value": "p00"
              }
            ]
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 2,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "advisor",
                "op": "eq",
                "value": "p00"
              },
              {
                "field": "year",
                "op": "gte",
                "value": "1"
              }
            ],
            "select": "id",
            "reduce": "list"
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 3,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "advisor",
                "op": "eq",
                "value": "p00"
              },
              {
                "field": "year",
                "op": "gte",
                "value": "1"
              }
            ],
            "select": "id",
            "reduce": "list"
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 4,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "direct_answer",
        "arguments": {
          "final_answer": "[\"student_id_1\", \"student_id_2\"]"
        }
      },
      "error": null,
      "invalid_arguments": false
    }
  ],
  "termination_reason": "direct_answer",
  "truncated": false,
  "number_of_steps": 4,
  "tool_sequence": [
    "search",
    "deep_search",
    "deep_search",
    "direct_answer"
  ],
  "token_ids": [],
  "action_token_mask": [],
  "logprobs": []
}
```

### Example 3: task-000 / rollout 0

```json
{
  "task_id": "task-000",
  "question": "What is the advisor ID of student s00?",
  "steps": [
    {
      "action": "deep_search",
      "tool_call_index": 0,
      "cost": 8
    },
    {
      "action": "lookup",
      "tool_call_index": 1,
      "cost": 1
    },
    {
      "action": "direct_answer",
      "arguments": {
        "final_answer": "[]"
      },
      "cost": 0
    }
  ],
  "tool_calls": [
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "students",
          "select": "id",
          "reduce": "list",
          "where": [
            {
              "field": "name",
              "op": "eq",
              "value": "s00"
            }
          ]
        }
      },
      "observation": {
        "answer": [],
        "results": [],
        "evidence": [
          {
            "entity": "s00",
            "field": "name",
            "value": "Student-00"
          },
          {
            "entity": "s01",
            "field": "name",
            "value": "Student-01"
          },
          {
            "entity": "s02",
            "field": "name",
            "value": "Student-02"
          },
          {
            "entity": "s03",
            "field": "name",
            "value": "Student-03"
          },
          {
            "entity": "s04",
            "field": "name",
            "value": "Student-04"
          },
          {
            "entity": "s05",
            "field": "name",
            "value": "Student-05"
          },
          {
            "entity": "s06",
            "field": "name",
            "value": "Student-06"
          },
          {
            "entity": "s07",
            "field": "name",
            "value": "Student-07"
          },
          {
            "entity": "s08",
            "field": "name",
            "value": "Student-08"
          },
          {
            "entity": "s09",
            "field": "name",
            "value": "Student-09"
          },
          {
            "entity": "s10",
            "field": "name",
            "value": "Student-10"
          },
          {
            "entity": "s11",
            "field": "name",
            "value": "Student-11"
          },
          {
            "entity": "s12",
            "field": "name",
            "value": "Student-12"
          },
          {
            "entity": "s13",
            "field": "name",
            "value": "Student-13"
          },
          {
            "entity": "s14",
            "field": "name",
            "value": "Student-14"
          },
          {
            "entity": "s15",
            "field": "name",
            "value": "Student-15"
          },
          {
            "entity": "s16",
            "field": "name",
            "value": "Student-16"
          },
          {
            "entity": "s17",
            "field": "name",
            "value": "Student-17"
          },
          {
            "entity": "s18",
            "field": "name",
            "value": "Student-18"
          },
          {
            "entity": "s19",
            "field": "name",
            "value": "Student-19"
          },
          {
            "entity": "s20",
            "field": "name",
            "value": "Student-20"
          },
          {
            "entity": "s21",
            "field": "name",
            "value": "Student-21"
          },
          {
            "entity": "s22",
            "field": "name",
            "value": "Student-22"
          },
          {
            "entity": "s23",
            "field": "name",
            "value": "Student-23"
          },
          {
            "entity": "s24",
            "field": "name",
            "value": "Student-24"
          },
          {
            "entity": "s25",
            "field": "name",
            "value": "Student-25"
          },
          {
            "entity": "s26",
            "field": "name",
            "value": "Student-26"
          },
          {
            "entity": "s27",
            "field": "name",
            "value": "Student-27"
          },
          {
            "entity": "s28",
            "field": "name",
            "value": "Student-28"
          },
          {
            "entity": "s29",
            "field": "name",
            "value": "Student-29"
          }
        ],
        "truncated": false,
        "matched_count": 0
      },
      "cost": 8
    },
    {
      "tool_name": "lookup",
      "arguments": {
        "entity": "s00",
        "field": "advisor"
      },
      "observation": {
        "value": "p03",
        "evidence": [
          {
            "entity": "s00",
            "field": "advisor",
            "value": "p03"
          }
        ]
      },
      "cost": 1
    }
  ],
  "observations": [
    {
      "answer": [],
      "results": [],
      "evidence": [
        {
          "entity": "s00",
          "field": "name",
          "value": "Student-00"
        },
        {
          "entity": "s01",
          "field": "name",
          "value": "Student-01"
        },
        {
          "entity": "s02",
          "field": "name",
          "value": "Student-02"
        },
        {
          "entity": "s03",
          "field": "name",
          "value": "Student-03"
        },
        {
          "entity": "s04",
          "field": "name",
          "value": "Student-04"
        },
        {
          "entity": "s05",
          "field": "name",
          "value": "Student-05"
        },
        {
          "entity": "s06",
          "field": "name",
          "value": "Student-06"
        },
        {
          "entity": "s07",
          "field": "name",
          "value": "Student-07"
        },
        {
          "entity": "s08",
          "field": "name",
          "value": "Student-08"
        },
        {
          "entity": "s09",
          "field": "name",
          "value": "Student-09"
        },
        {
          "entity": "s10",
          "field": "name",
          "value": "Student-10"
        },
        {
          "entity": "s11",
          "field": "name",
          "value": "Student-11"
        },
        {
          "entity": "s12",
          "field": "name",
          "value": "Student-12"
        },
        {
          "entity": "s13",
          "field": "name",
          "value": "Student-13"
        },
        {
          "entity": "s14",
          "field": "name",
          "value": "Student-14"
        },
        {
          "entity": "s15",
          "field": "name",
          "value": "Student-15"
        },
        {
          "entity": "s16",
          "field": "name",
          "value": "Student-16"
        },
        {
          "entity": "s17",
          "field": "name",
          "value": "Student-17"
        },
        {
          "entity": "s18",
          "field": "name",
          "value": "Student-18"
        },
        {
          "entity": "s19",
          "field": "name",
          "value": "Student-19"
        },
        {
          "entity": "s20",
          "field": "name",
          "value": "Student-20"
        },
        {
          "entity": "s21",
          "field": "name",
          "value": "Student-21"
        },
        {
          "entity": "s22",
          "field": "name",
          "value": "Student-22"
        },
        {
          "entity": "s23",
          "field": "name",
          "value": "Student-23"
        },
        {
          "entity": "s24",
          "field": "name",
          "value": "Student-24"
        },
        {
          "entity": "s25",
          "field": "name",
          "value": "Student-25"
        },
        {
          "entity": "s26",
          "field": "name",
          "value": "Student-26"
        },
        {
          "entity": "s27",
          "field": "name",
          "value": "Student-27"
        },
        {
          "entity": "s28",
          "field": "name",
          "value": "Student-28"
        },
        {
          "entity": "s29",
          "field": "name",
          "value": "Student-29"
        }
      ],
      "truncated": false,
      "matched_count": 0
    },
    {
      "value": "p03",
      "evidence": [
        {
          "entity": "s00",
          "field": "advisor",
          "value": "p03"
        }
      ]
    }
  ],
  "final_answer": "[]",
  "total_cost": 9,
  "metadata": {
    "rollout_id": 0,
    "rollout_config": {
      "max_steps": 4,
      "max_tool_calls": 3,
      "max_parse_retries": 1,
      "seed": 2026
    },
    "runtime": {
      "device": "cpu",
      "dtype": "float32",
      "torch_version": "2.8.0",
      "transformers_version": "4.57.6",
      "resolved_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    }
  },
  "policy_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "sampling_config": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 192,
    "do_sample": true,
    "seed": 6022803196224264726
  },
  "raw_outputs": [
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"students\", \"select\": \"id\", \"reduce\": \"list\", \"where\": [{\"field\": \"name\", \"op\": \"eq\", \"value\": \"s00\"}]}}}",
    "{\"action\": \"lookup\", \"arguments\": {\"entity\": \"s00\", \"field\": \"advisor\"}}",
    "{\"action\": \"direct_answer\", \"arguments\": {\"final_answer\": \"[]\"}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "select": "id",
            "reduce": "list",
            "where": [
              {
                "field": "name",
                "op": "eq",
                "value": "s00"
              }
            ]
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 2,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "lookup",
        "arguments": {
          "entity": "s00",
          "field": "advisor"
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 3,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "direct_answer",
        "arguments": {
          "final_answer": "[]"
        }
      },
      "error": null,
      "invalid_arguments": false
    }
  ],
  "termination_reason": "direct_answer",
  "truncated": false,
  "number_of_steps": 3,
  "tool_sequence": [
    "deep_search",
    "lookup",
    "direct_answer"
  ],
  "token_ids": [],
  "action_token_mask": [],
  "logprobs": []
}
```

### Example 4: task-001 / rollout 0

```json
{
  "task_id": "task-001",
  "question": "What is the department of the advisor of student s00?",
  "steps": [
    {
      "action": "deep_search",
      "tool_call_index": 0,
      "cost": 8
    },
    {
      "action": "lookup",
      "tool_call_index": 1,
      "cost": 1
    },
    {
      "action": "invalid_action",
      "raw_output_indices": [
        3,
        4
      ],
      "cost": 0
    }
  ],
  "tool_calls": [
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "professors",
          "where": [
            {
              "field": "name",
              "op": "contains",
              "value": "advisor"
            }
          ],
          "select": "department"
        }
      },
      "observation": {
        "answer": [],
        "results": [],
        "evidence": [
          {
            "entity": "p00",
            "field": "name",
            "value": "Professor-00"
          },
          {
            "entity": "p01",
            "field": "name",
            "value": "Professor-01"
          },
          {
            "entity": "p02",
            "field": "name",
            "value": "Professor-02"
          },
          {
            "entity": "p03",
            "field": "name",
            "value": "Professor-03"
          },
          {
            "entity": "p04",
            "field": "name",
            "value": "Professor-04"
          }
        ],
        "truncated": false,
        "matched_count": 0
      },
      "cost": 8
    },
    {
      "tool_name": "lookup",
      "arguments": {
        "entity": "s00",
        "field": "name"
      },
      "observation": {
        "value": "Student-00",
        "evidence": [
          {
            "entity": "s00",
            "field": "name",
            "value": "Student-00"
          }
        ]
      },
      "cost": 1
    }
  ],
  "observations": [
    {
      "answer": [],
      "results": [],
      "evidence": [
        {
          "entity": "p00",
          "field": "name",
          "value": "Professor-00"
        },
        {
          "entity": "p01",
          "field": "name",
          "value": "Professor-01"
        },
        {
          "entity": "p02",
          "field": "name",
          "value": "Professor-02"
        },
        {
          "entity": "p03",
          "field": "name",
          "value": "Professor-03"
        },
        {
          "entity": "p04",
          "field": "name",
          "value": "Professor-04"
        }
      ],
      "truncated": false,
      "matched_count": 0
    },
    {
      "value": "Student-00",
      "evidence": [
        {
          "entity": "s00",
          "field": "name",
          "value": "Student-00"
        }
      ]
    }
  ],
  "final_answer": null,
  "total_cost": 9,
  "metadata": {
    "rollout_id": 0,
    "rollout_config": {
      "max_steps": 4,
      "max_tool_calls": 3,
      "max_parse_retries": 1,
      "seed": 2026
    },
    "runtime": {
      "device": "cpu",
      "dtype": "float32",
      "torch_version": "2.8.0",
      "transformers_version": "4.57.6",
      "resolved_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    }
  },
  "policy_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "sampling_config": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 192,
    "do_sample": true,
    "seed": 7331016465707914508
  },
  "raw_outputs": [
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"professors\", \"where\": [{\"field\": \"name\", \"op\": \"contains\", \"value\": \"advisor\"}], \"select\": \"department\"}}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"name\", \"op\": \"equals\", \"value\": \"s00\"}]}}}",
    "{\"action\":\"lookup\",\"arguments\":{\"entity\":\"s00\",\"field\":\"name\"}}",
    "{\"action\": \"search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"name\", \"op\": \"equals\", \"value\": \"s00\"}], \"select\": \"advisor_id\", \"reduce\": \"list\"}}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"advisors\", \"where\": [{\"field\": \"student\", \"op\": \"eq\", \"value\": \"s00\"}], \"select\": \"department\", \"reduce\": \"list\"}}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "professors",
            "where": [
              {
                "field": "name",
                "op": "contains",
                "value": "advisor"
              }
            ],
            "select": "department"
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 2,
      "retry_count": 0,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "name",
                "op": "equals",
                "value": "s00"
              }
            ]
          }
        }
      },
      "error": "Invalid operator",
      "invalid_arguments": true
    },
    {
      "step": 2,
      "retry_count": 1,
      "success": true,
      "parsed_action": {
        "action": "lookup",
        "arguments": {
          "entity": "s00",
          "field": "name"
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 3,
      "retry_count": 0,
      "success": false,
      "parsed_action": {
        "action": "search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "name",
                "op": "equals",
                "value": "s00"
              }
            ],
            "select": "advisor_id",
            "reduce": "list"
          }
        }
      },
      "error": "Unknown field for table: advisor_id",
      "invalid_arguments": true
    },
    {
      "step": 3,
      "retry_count": 1,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "advisors",
            "where": [
              {
                "field": "student",
                "op": "eq",
                "value": "s00"
              }
            ],
            "select": "department",
            "reduce": "list"
          }
        }
      },
      "error": "Unknown table",
      "invalid_arguments": true
    }
  ],
  "termination_reason": "invalid_action",
  "truncated": false,
  "number_of_steps": 3,
  "tool_sequence": [
    "deep_search",
    "lookup"
  ],
  "token_ids": [],
  "action_token_mask": [],
  "logprobs": []
}
```

### Example 5: task-032 / rollout 1

```json
{
  "task_id": "task-032",
  "question": "List student IDs in year 1.",
  "steps": [
    {
      "action": "deep_search",
      "tool_call_index": 0,
      "cost": 8
    },
    {
      "action": "deep_search",
      "tool_call_index": 1,
      "cost": 8
    },
    {
      "action": "search",
      "tool_call_index": 2,
      "cost": 2
    },
    {
      "action": "blocked_tool_call",
      "proposed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "gte",
                "value": 1
              }
            ],
            "select": "id"
          }
        }
      },
      "cost": 0
    }
  ],
  "tool_calls": [
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "year",
              "op": "gte",
              "value": 1
            }
          ]
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02",
          "s03",
          "s04",
          "s05",
          "s06",
          "s07",
          "s08",
          "s09",
          "s10",
          "s11",
          "s12",
          "s13",
          "s14",
          "s15",
          "s16",
          "s17",
          "s18",
          "s19",
          "s20",
          "s21",
          "s22",
          "s23",
          "s24",
          "s25",
          "s26",
          "s27",
          "s28",
          "s29"
        ],
        "results": [
          {
            "entity": "s00",
            "values": [
              "s00"
            ]
          },
          {
            "entity": "s01",
            "values": [
              "s01"
            ]
          },
          {
            "entity": "s02",
            "values": [
              "s02"
            ]
          },
          {
            "entity": "s03",
            "values": [
              "s03"
            ]
          },
          {
            "entity": "s04",
            "values": [
              "s04"
            ]
          },
          {
            "entity": "s05",
            "values": [
              "s05"
            ]
          },
          {
            "entity": "s06",
            "values": [
              "s06"
            ]
          },
          {
            "entity": "s07",
            "values": [
              "s07"
            ]
          },
          {
            "entity": "s08",
            "values": [
              "s08"
            ]
          },
          {
            "entity": "s09",
            "values": [
              "s09"
            ]
          },
          {
            "entity": "s10",
            "values": [
              "s10"
            ]
          },
          {
            "entity": "s11",
            "values": [
              "s11"
            ]
          },
          {
            "entity": "s12",
            "values": [
              "s12"
            ]
          },
          {
            "entity": "s13",
            "values": [
              "s13"
            ]
          },
          {
            "entity": "s14",
            "values": [
              "s14"
            ]
          },
          {
            "entity": "s15",
            "values": [
              "s15"
            ]
          },
          {
            "entity": "s16",
            "values": [
              "s16"
            ]
          },
          {
            "entity": "s17",
            "values": [
              "s17"
            ]
          },
          {
            "entity": "s18",
            "values": [
              "s18"
            ]
          },
          {
            "entity": "s19",
            "values": [
              "s19"
            ]
          },
          {
            "entity": "s20",
            "values": [
              "s20"
            ]
          },
          {
            "entity": "s21",
            "values": [
              "s21"
            ]
          },
          {
            "entity": "s22",
            "values": [
              "s22"
            ]
          },
          {
            "entity": "s23",
            "values": [
              "s23"
            ]
          },
          {
            "entity": "s24",
            "values": [
              "s24"
            ]
          },
          {
            "entity": "s25",
            "values": [
              "s25"
            ]
          },
          {
            "entity": "s26",
            "values": [
              "s26"
            ]
          },
          {
            "entity": "s27",
            "values": [
              "s27"
            ]
          },
          {
            "entity": "s28",
            "values": [
              "s28"
            ]
          },
          {
            "entity": "s29",
            "values": [
              "s29"
            ]
          }
        ],
        "evidence": [
          {
            "entity": "s00",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s00",
            "field": "id",
            "value": "s00"
          },
          {
            "entity": "s01",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s01",
            "field": "id",
            "value": "s01"
          },
          {
            "entity": "s02",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s02",
            "field": "id",
            "value": "s02"
          },
          {
            "entity": "s03",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s03",
            "field": "id",
            "value": "s03"
          },
          {
            "entity": "s04",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s04",
            "field": "id",
            "value": "s04"
          },
          {
            "entity": "s05",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s05",
            "field": "id",
            "value": "s05"
          },
          {
            "entity": "s06",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s06",
            "field": "id",
            "value": "s06"
          },
          {
            "entity": "s07",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s07",
            "field": "id",
            "value": "s07"
          },
          {
            "entity": "s08",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s08",
            "field": "id",
            "value": "s08"
          },
          {
            "entity": "s09",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s09",
            "field": "id",
            "value": "s09"
          },
          {
            "entity": "s10",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s10",
            "field": "id",
            "value": "s10"
          },
          {
            "entity": "s11",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s11",
            "field": "id",
            "value": "s11"
          },
          {
            "entity": "s12",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s12",
            "field": "id",
            "value": "s12"
          },
          {
            "entity": "s13",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s13",
            "field": "id",
            "value": "s13"
          },
          {
            "entity": "s14",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s14",
            "field": "id",
            "value": "s14"
          },
          {
            "entity": "s15",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s15",
            "field": "id",
            "value": "s15"
          },
          {
            "entity": "s16",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s16",
            "field": "id",
            "value": "s16"
          },
          {
            "entity": "s17",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s17",
            "field": "id",
            "value": "s17"
          },
          {
            "entity": "s18",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s18",
            "field": "id",
            "value": "s18"
          },
          {
            "entity": "s19",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s19",
            "field": "id",
            "value": "s19"
          },
          {
            "entity": "s20",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s20",
            "field": "id",
            "value": "s20"
          },
          {
            "entity": "s21",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s21",
            "field": "id",
            "value": "s21"
          },
          {
            "entity": "s22",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s22",
            "field": "id",
            "value": "s22"
          },
          {
            "entity": "s23",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s23",
            "field": "id",
            "value": "s23"
          },
          {
            "entity": "s24",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s24",
            "field": "id",
            "value": "s24"
          },
          {
            "entity": "s25",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s25",
            "field": "id",
            "value": "s25"
          },
          {
            "entity": "s26",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s26",
            "field": "id",
            "value": "s26"
          },
          {
            "entity": "s27",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s27",
            "field": "id",
            "value": "s27"
          },
          {
            "entity": "s28",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s28",
            "field": "id",
            "value": "s28"
          },
          {
            "entity": "s29",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s29",
            "field": "id",
            "value": "s29"
          }
        ],
        "truncated": false,
        "matched_count": 30
      },
      "cost": 8
    },
    {
      "tool_name": "deep_search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "year",
              "op": "gte",
              "value": 1
            }
          ],
          "select": "id"
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02",
          "s03",
          "s04",
          "s05",
          "s06",
          "s07",
          "s08",
          "s09",
          "s10",
          "s11",
          "s12",
          "s13",
          "s14",
          "s15",
          "s16",
          "s17",
          "s18",
          "s19",
          "s20",
          "s21",
          "s22",
          "s23",
          "s24",
          "s25",
          "s26",
          "s27",
          "s28",
          "s29"
        ],
        "results": [
          {
            "entity": "s00",
            "values": [
              "s00"
            ]
          },
          {
            "entity": "s01",
            "values": [
              "s01"
            ]
          },
          {
            "entity": "s02",
            "values": [
              "s02"
            ]
          },
          {
            "entity": "s03",
            "values": [
              "s03"
            ]
          },
          {
            "entity": "s04",
            "values": [
              "s04"
            ]
          },
          {
            "entity": "s05",
            "values": [
              "s05"
            ]
          },
          {
            "entity": "s06",
            "values": [
              "s06"
            ]
          },
          {
            "entity": "s07",
            "values": [
              "s07"
            ]
          },
          {
            "entity": "s08",
            "values": [
              "s08"
            ]
          },
          {
            "entity": "s09",
            "values": [
              "s09"
            ]
          },
          {
            "entity": "s10",
            "values": [
              "s10"
            ]
          },
          {
            "entity": "s11",
            "values": [
              "s11"
            ]
          },
          {
            "entity": "s12",
            "values": [
              "s12"
            ]
          },
          {
            "entity": "s13",
            "values": [
              "s13"
            ]
          },
          {
            "entity": "s14",
            "values": [
              "s14"
            ]
          },
          {
            "entity": "s15",
            "values": [
              "s15"
            ]
          },
          {
            "entity": "s16",
            "values": [
              "s16"
            ]
          },
          {
            "entity": "s17",
            "values": [
              "s17"
            ]
          },
          {
            "entity": "s18",
            "values": [
              "s18"
            ]
          },
          {
            "entity": "s19",
            "values": [
              "s19"
            ]
          },
          {
            "entity": "s20",
            "values": [
              "s20"
            ]
          },
          {
            "entity": "s21",
            "values": [
              "s21"
            ]
          },
          {
            "entity": "s22",
            "values": [
              "s22"
            ]
          },
          {
            "entity": "s23",
            "values": [
              "s23"
            ]
          },
          {
            "entity": "s24",
            "values": [
              "s24"
            ]
          },
          {
            "entity": "s25",
            "values": [
              "s25"
            ]
          },
          {
            "entity": "s26",
            "values": [
              "s26"
            ]
          },
          {
            "entity": "s27",
            "values": [
              "s27"
            ]
          },
          {
            "entity": "s28",
            "values": [
              "s28"
            ]
          },
          {
            "entity": "s29",
            "values": [
              "s29"
            ]
          }
        ],
        "evidence": [
          {
            "entity": "s00",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s00",
            "field": "id",
            "value": "s00"
          },
          {
            "entity": "s01",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s01",
            "field": "id",
            "value": "s01"
          },
          {
            "entity": "s02",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s02",
            "field": "id",
            "value": "s02"
          },
          {
            "entity": "s03",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s03",
            "field": "id",
            "value": "s03"
          },
          {
            "entity": "s04",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s04",
            "field": "id",
            "value": "s04"
          },
          {
            "entity": "s05",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s05",
            "field": "id",
            "value": "s05"
          },
          {
            "entity": "s06",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s06",
            "field": "id",
            "value": "s06"
          },
          {
            "entity": "s07",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s07",
            "field": "id",
            "value": "s07"
          },
          {
            "entity": "s08",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s08",
            "field": "id",
            "value": "s08"
          },
          {
            "entity": "s09",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s09",
            "field": "id",
            "value": "s09"
          },
          {
            "entity": "s10",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s10",
            "field": "id",
            "value": "s10"
          },
          {
            "entity": "s11",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s11",
            "field": "id",
            "value": "s11"
          },
          {
            "entity": "s12",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s12",
            "field": "id",
            "value": "s12"
          },
          {
            "entity": "s13",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s13",
            "field": "id",
            "value": "s13"
          },
          {
            "entity": "s14",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s14",
            "field": "id",
            "value": "s14"
          },
          {
            "entity": "s15",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s15",
            "field": "id",
            "value": "s15"
          },
          {
            "entity": "s16",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s16",
            "field": "id",
            "value": "s16"
          },
          {
            "entity": "s17",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s17",
            "field": "id",
            "value": "s17"
          },
          {
            "entity": "s18",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s18",
            "field": "id",
            "value": "s18"
          },
          {
            "entity": "s19",
            "field": "year",
            "value": 3
          },
          {
            "entity": "s19",
            "field": "id",
            "value": "s19"
          },
          {
            "entity": "s20",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s20",
            "field": "id",
            "value": "s20"
          },
          {
            "entity": "s21",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s21",
            "field": "id",
            "value": "s21"
          },
          {
            "entity": "s22",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s22",
            "field": "id",
            "value": "s22"
          },
          {
            "entity": "s23",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s23",
            "field": "id",
            "value": "s23"
          },
          {
            "entity": "s24",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s24",
            "field": "id",
            "value": "s24"
          },
          {
            "entity": "s25",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s25",
            "field": "id",
            "value": "s25"
          },
          {
            "entity": "s26",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s26",
            "field": "id",
            "value": "s26"
          },
          {
            "entity": "s27",
            "field": "year",
            "value": 2
          },
          {
            "entity": "s27",
            "field": "id",
            "value": "s27"
          },
          {
            "entity": "s28",
            "field": "year",
            "value": 1
          },
          {
            "entity": "s28",
            "field": "id",
            "value": "s28"
          },
          {
            "entity": "s29",
            "field": "year",
            "value": 4
          },
          {
            "entity": "s29",
            "field": "id",
            "value": "s29"
          }
        ],
        "truncated": false,
        "matched_count": 30
      },
      "cost": 8
    },
    {
      "tool_name": "search",
      "arguments": {
        "query": {
          "table": "students",
          "where": [
            {
              "field": "year",
              "op": "gte",
              "value": 1
            }
          ]
        }
      },
      "observation": {
        "answer": [
          "s00",
          "s01",
          "s02",
          "s03",
          "s04"
        ],
        "results": [
          {
            "entity": "s00",
            "values": [
              "s00"
            ]
          },
          {
            "entity": "s01",
            "values": [
              "s01"
            ]
          },
          {
            "entity": "s02",
            "values": [
              "s02"
            ]
          },
          {
            "entity": "s03",
            "values": [
              "s03"
            ]
          },
          {
            "entity": "s04",
            "values": [
              "s04"
            ]
          }
        ],
        "truncated": true,
        "matched_count": 30
      },
      "cost": 2
    }
  ],
  "observations": [
    {
      "answer": [
        "s00",
        "s01",
        "s02",
        "s03",
        "s04",
        "s05",
        "s06",
        "s07",
        "s08",
        "s09",
        "s10",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
        "s16",
        "s17",
        "s18",
        "s19",
        "s20",
        "s21",
        "s22",
        "s23",
        "s24",
        "s25",
        "s26",
        "s27",
        "s28",
        "s29"
      ],
      "results": [
        {
          "entity": "s00",
          "values": [
            "s00"
          ]
        },
        {
          "entity": "s01",
          "values": [
            "s01"
          ]
        },
        {
          "entity": "s02",
          "values": [
            "s02"
          ]
        },
        {
          "entity": "s03",
          "values": [
            "s03"
          ]
        },
        {
          "entity": "s04",
          "values": [
            "s04"
          ]
        },
        {
          "entity": "s05",
          "values": [
            "s05"
          ]
        },
        {
          "entity": "s06",
          "values": [
            "s06"
          ]
        },
        {
          "entity": "s07",
          "values": [
            "s07"
          ]
        },
        {
          "entity": "s08",
          "values": [
            "s08"
          ]
        },
        {
          "entity": "s09",
          "values": [
            "s09"
          ]
        },
        {
          "entity": "s10",
          "values": [
            "s10"
          ]
        },
        {
          "entity": "s11",
          "values": [
            "s11"
          ]
        },
        {
          "entity": "s12",
          "values": [
            "s12"
          ]
        },
        {
          "entity": "s13",
          "values": [
            "s13"
          ]
        },
        {
          "entity": "s14",
          "values": [
            "s14"
          ]
        },
        {
          "entity": "s15",
          "values": [
            "s15"
          ]
        },
        {
          "entity": "s16",
          "values": [
            "s16"
          ]
        },
        {
          "entity": "s17",
          "values": [
            "s17"
          ]
        },
        {
          "entity": "s18",
          "values": [
            "s18"
          ]
        },
        {
          "entity": "s19",
          "values": [
            "s19"
          ]
        },
        {
          "entity": "s20",
          "values": [
            "s20"
          ]
        },
        {
          "entity": "s21",
          "values": [
            "s21"
          ]
        },
        {
          "entity": "s22",
          "values": [
            "s22"
          ]
        },
        {
          "entity": "s23",
          "values": [
            "s23"
          ]
        },
        {
          "entity": "s24",
          "values": [
            "s24"
          ]
        },
        {
          "entity": "s25",
          "values": [
            "s25"
          ]
        },
        {
          "entity": "s26",
          "values": [
            "s26"
          ]
        },
        {
          "entity": "s27",
          "values": [
            "s27"
          ]
        },
        {
          "entity": "s28",
          "values": [
            "s28"
          ]
        },
        {
          "entity": "s29",
          "values": [
            "s29"
          ]
        }
      ],
      "evidence": [
        {
          "entity": "s00",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s00",
          "field": "id",
          "value": "s00"
        },
        {
          "entity": "s01",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s01",
          "field": "id",
          "value": "s01"
        },
        {
          "entity": "s02",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s02",
          "field": "id",
          "value": "s02"
        },
        {
          "entity": "s03",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s03",
          "field": "id",
          "value": "s03"
        },
        {
          "entity": "s04",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s04",
          "field": "id",
          "value": "s04"
        },
        {
          "entity": "s05",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s05",
          "field": "id",
          "value": "s05"
        },
        {
          "entity": "s06",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s06",
          "field": "id",
          "value": "s06"
        },
        {
          "entity": "s07",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s07",
          "field": "id",
          "value": "s07"
        },
        {
          "entity": "s08",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s08",
          "field": "id",
          "value": "s08"
        },
        {
          "entity": "s09",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s09",
          "field": "id",
          "value": "s09"
        },
        {
          "entity": "s10",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s10",
          "field": "id",
          "value": "s10"
        },
        {
          "entity": "s11",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s11",
          "field": "id",
          "value": "s11"
        },
        {
          "entity": "s12",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s12",
          "field": "id",
          "value": "s12"
        },
        {
          "entity": "s13",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s13",
          "field": "id",
          "value": "s13"
        },
        {
          "entity": "s14",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s14",
          "field": "id",
          "value": "s14"
        },
        {
          "entity": "s15",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s15",
          "field": "id",
          "value": "s15"
        },
        {
          "entity": "s16",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s16",
          "field": "id",
          "value": "s16"
        },
        {
          "entity": "s17",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s17",
          "field": "id",
          "value": "s17"
        },
        {
          "entity": "s18",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s18",
          "field": "id",
          "value": "s18"
        },
        {
          "entity": "s19",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s19",
          "field": "id",
          "value": "s19"
        },
        {
          "entity": "s20",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s20",
          "field": "id",
          "value": "s20"
        },
        {
          "entity": "s21",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s21",
          "field": "id",
          "value": "s21"
        },
        {
          "entity": "s22",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s22",
          "field": "id",
          "value": "s22"
        },
        {
          "entity": "s23",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s23",
          "field": "id",
          "value": "s23"
        },
        {
          "entity": "s24",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s24",
          "field": "id",
          "value": "s24"
        },
        {
          "entity": "s25",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s25",
          "field": "id",
          "value": "s25"
        },
        {
          "entity": "s26",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s26",
          "field": "id",
          "value": "s26"
        },
        {
          "entity": "s27",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s27",
          "field": "id",
          "value": "s27"
        },
        {
          "entity": "s28",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s28",
          "field": "id",
          "value": "s28"
        },
        {
          "entity": "s29",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s29",
          "field": "id",
          "value": "s29"
        }
      ],
      "truncated": false,
      "matched_count": 30
    },
    {
      "answer": [
        "s00",
        "s01",
        "s02",
        "s03",
        "s04",
        "s05",
        "s06",
        "s07",
        "s08",
        "s09",
        "s10",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
        "s16",
        "s17",
        "s18",
        "s19",
        "s20",
        "s21",
        "s22",
        "s23",
        "s24",
        "s25",
        "s26",
        "s27",
        "s28",
        "s29"
      ],
      "results": [
        {
          "entity": "s00",
          "values": [
            "s00"
          ]
        },
        {
          "entity": "s01",
          "values": [
            "s01"
          ]
        },
        {
          "entity": "s02",
          "values": [
            "s02"
          ]
        },
        {
          "entity": "s03",
          "values": [
            "s03"
          ]
        },
        {
          "entity": "s04",
          "values": [
            "s04"
          ]
        },
        {
          "entity": "s05",
          "values": [
            "s05"
          ]
        },
        {
          "entity": "s06",
          "values": [
            "s06"
          ]
        },
        {
          "entity": "s07",
          "values": [
            "s07"
          ]
        },
        {
          "entity": "s08",
          "values": [
            "s08"
          ]
        },
        {
          "entity": "s09",
          "values": [
            "s09"
          ]
        },
        {
          "entity": "s10",
          "values": [
            "s10"
          ]
        },
        {
          "entity": "s11",
          "values": [
            "s11"
          ]
        },
        {
          "entity": "s12",
          "values": [
            "s12"
          ]
        },
        {
          "entity": "s13",
          "values": [
            "s13"
          ]
        },
        {
          "entity": "s14",
          "values": [
            "s14"
          ]
        },
        {
          "entity": "s15",
          "values": [
            "s15"
          ]
        },
        {
          "entity": "s16",
          "values": [
            "s16"
          ]
        },
        {
          "entity": "s17",
          "values": [
            "s17"
          ]
        },
        {
          "entity": "s18",
          "values": [
            "s18"
          ]
        },
        {
          "entity": "s19",
          "values": [
            "s19"
          ]
        },
        {
          "entity": "s20",
          "values": [
            "s20"
          ]
        },
        {
          "entity": "s21",
          "values": [
            "s21"
          ]
        },
        {
          "entity": "s22",
          "values": [
            "s22"
          ]
        },
        {
          "entity": "s23",
          "values": [
            "s23"
          ]
        },
        {
          "entity": "s24",
          "values": [
            "s24"
          ]
        },
        {
          "entity": "s25",
          "values": [
            "s25"
          ]
        },
        {
          "entity": "s26",
          "values": [
            "s26"
          ]
        },
        {
          "entity": "s27",
          "values": [
            "s27"
          ]
        },
        {
          "entity": "s28",
          "values": [
            "s28"
          ]
        },
        {
          "entity": "s29",
          "values": [
            "s29"
          ]
        }
      ],
      "evidence": [
        {
          "entity": "s00",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s00",
          "field": "id",
          "value": "s00"
        },
        {
          "entity": "s01",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s01",
          "field": "id",
          "value": "s01"
        },
        {
          "entity": "s02",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s02",
          "field": "id",
          "value": "s02"
        },
        {
          "entity": "s03",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s03",
          "field": "id",
          "value": "s03"
        },
        {
          "entity": "s04",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s04",
          "field": "id",
          "value": "s04"
        },
        {
          "entity": "s05",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s05",
          "field": "id",
          "value": "s05"
        },
        {
          "entity": "s06",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s06",
          "field": "id",
          "value": "s06"
        },
        {
          "entity": "s07",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s07",
          "field": "id",
          "value": "s07"
        },
        {
          "entity": "s08",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s08",
          "field": "id",
          "value": "s08"
        },
        {
          "entity": "s09",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s09",
          "field": "id",
          "value": "s09"
        },
        {
          "entity": "s10",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s10",
          "field": "id",
          "value": "s10"
        },
        {
          "entity": "s11",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s11",
          "field": "id",
          "value": "s11"
        },
        {
          "entity": "s12",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s12",
          "field": "id",
          "value": "s12"
        },
        {
          "entity": "s13",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s13",
          "field": "id",
          "value": "s13"
        },
        {
          "entity": "s14",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s14",
          "field": "id",
          "value": "s14"
        },
        {
          "entity": "s15",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s15",
          "field": "id",
          "value": "s15"
        },
        {
          "entity": "s16",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s16",
          "field": "id",
          "value": "s16"
        },
        {
          "entity": "s17",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s17",
          "field": "id",
          "value": "s17"
        },
        {
          "entity": "s18",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s18",
          "field": "id",
          "value": "s18"
        },
        {
          "entity": "s19",
          "field": "year",
          "value": 3
        },
        {
          "entity": "s19",
          "field": "id",
          "value": "s19"
        },
        {
          "entity": "s20",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s20",
          "field": "id",
          "value": "s20"
        },
        {
          "entity": "s21",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s21",
          "field": "id",
          "value": "s21"
        },
        {
          "entity": "s22",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s22",
          "field": "id",
          "value": "s22"
        },
        {
          "entity": "s23",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s23",
          "field": "id",
          "value": "s23"
        },
        {
          "entity": "s24",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s24",
          "field": "id",
          "value": "s24"
        },
        {
          "entity": "s25",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s25",
          "field": "id",
          "value": "s25"
        },
        {
          "entity": "s26",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s26",
          "field": "id",
          "value": "s26"
        },
        {
          "entity": "s27",
          "field": "year",
          "value": 2
        },
        {
          "entity": "s27",
          "field": "id",
          "value": "s27"
        },
        {
          "entity": "s28",
          "field": "year",
          "value": 1
        },
        {
          "entity": "s28",
          "field": "id",
          "value": "s28"
        },
        {
          "entity": "s29",
          "field": "year",
          "value": 4
        },
        {
          "entity": "s29",
          "field": "id",
          "value": "s29"
        }
      ],
      "truncated": false,
      "matched_count": 30
    },
    {
      "answer": [
        "s00",
        "s01",
        "s02",
        "s03",
        "s04"
      ],
      "results": [
        {
          "entity": "s00",
          "values": [
            "s00"
          ]
        },
        {
          "entity": "s01",
          "values": [
            "s01"
          ]
        },
        {
          "entity": "s02",
          "values": [
            "s02"
          ]
        },
        {
          "entity": "s03",
          "values": [
            "s03"
          ]
        },
        {
          "entity": "s04",
          "values": [
            "s04"
          ]
        }
      ],
      "truncated": true,
      "matched_count": 30
    }
  ],
  "final_answer": null,
  "total_cost": 18,
  "metadata": {
    "rollout_id": 1,
    "rollout_config": {
      "max_steps": 4,
      "max_tool_calls": 3,
      "max_parse_retries": 1,
      "seed": 2026
    },
    "runtime": {
      "device": "cpu",
      "dtype": "float32",
      "torch_version": "2.8.0",
      "transformers_version": "4.57.6",
      "resolved_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    }
  },
  "policy_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "sampling_config": {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 192,
    "do_sample": true,
    "seed": 69266093609787937
  },
  "raw_outputs": [
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"year\", \"op\": \"gte\", \"value\": 1}]}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"table\":\"students\",\"where\":[{\"field\":\"year\",\"op\":\"gte\",\"value\":1}],\"select\":\"id\"}}}",
    "{\"action\": \"search\", \"arguments\": {\"query\": {\"table\": \"students\", \"where\": [{\"field\": \"year\", \"op\": \"gte\", \"value\": 1}]}}}",
    "{\"action\":\"deep_search\",\"arguments\":{\"query\":{\"table\":\"students\",\"where\":[{\"field\":\"year\",\"op\":\"gte\",\"value\":1}],\"select\":\"id\"}}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "gte",
                "value": 1
              }
            ]
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 2,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "gte",
                "value": 1
              }
            ],
            "select": "id"
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 3,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "gte",
                "value": 1
              }
            ]
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    },
    {
      "step": 4,
      "retry_count": 0,
      "success": true,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": {
            "table": "students",
            "where": [
              {
                "field": "year",
                "op": "gte",
                "value": 1
              }
            ],
            "select": "id"
          }
        }
      },
      "error": null,
      "invalid_arguments": false
    }
  ],
  "termination_reason": "max_tool_calls",
  "truncated": true,
  "number_of_steps": 4,
  "tool_sequence": [
    "deep_search",
    "deep_search",
    "search"
  ],
  "token_ids": [],
  "action_token_mask": [],
  "logprobs": []
}
```


## 本次真实 CPU 结果解读

本次使用固定 revision 的 Qwen2.5-1.5B-Instruct，4 道题 × G=2，共 8 条；这不是默认的 12 道题 × G=4 CUDA 正式实验。初版 prompt 的另外 8 条失败记录另存，不并入下面的统计。

- 准确率 1/8（12.5%）；6 条提交答案、1 条工具预算截断、1 条最终解析失败。平均成本 12.875，平均 2.125 次工具调用。
- 每条轨迹均实际使用 deep_search；17 次工具调用中 deep_search 12 次（70.59%）、lookup 3 次（17.65%）、search 2 次（11.76%）。首个提出的 action 为 deep_search 7 次、search 1 次，没有直接回答。
- 27 次模型输出中有 3 次解析/schema 失败（11.11%，含 repair）。另有 3 次已执行工具的类型错误，未混入解析错误统计。
- 四道题各自的两条 action 序列均不相同；成本方差分别为 task-032: 1、task-033: 1、task-000: 0.25、task-001: 0。仅 G=2，不足以估计稳定的分布。
- 两条轨迹出现完全相同的重复工具调用；五条最终答案是列表样式的字符串。模型还混淆 name 与 id，并在收到数值/字符串比较错误后重复错误参数。
- 单跳题 task-000 的 rollout 0 经 lookup 已拿到 p03，却提交字符串 "[]"，说明成功取证不保证最终使用证据。
- 唯一按答案判对的 task-001 rollout 1 也不能视为正确的推理链：模型查询的是 p00 的 department，而 s00 的真实 advisor 是 p03；两者 department 恰好都是 Ecology。答案准确率无法检出这种错误关联，后续 AI Judge 需要单独审查 evidence 是否支持题目所问的关系。

这批数据可验证真实 rollout、失败记录和组内差异。由于样本极小、prompt 已因接口格式修订过，且正确率较低，不应据此声称存在稳定的模型工具偏好或成本无感导致的过度使用。

初版 prompt 记录见 `../data/cpu_smoke_initial_prompt/`，最终原始记录见 `../data/cpu_smoke/policy_rollouts.jsonl`，额外完整示例见 `../data/cpu_smoke/example_trajectory.json`。默认 CUDA 配置尚未在本机执行；本机无 CUDA。
