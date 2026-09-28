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
  "elapsed_seconds": 92.87887287139893,
  "task_ids": [
    "task-032",
    "task-033",
    "task-000",
    "task-001"
  ],
  "kb_sha256": "fa01f5ceded2f75eb053df2045c147d48c0376dc1aee4ef22e44512bf3283775",
  "public_tasks_sha256": "83cf8ea10d1b3643805a14443c2ec7566185e86674d718c7e65194cae1064c23",
  "evaluation_tasks_sha256": "158385f6cf1afd615d65d0dca9f4656cf17e8210559f8ab3be7a340b00b139a9",
  "prompt_sha256": "b57570547ef456b81c159ea6a04deae97ea69f1b7b59e82d6c55ce8ca6f81ec8",
  "archive_note": "Initial prompt pilot. Preserved before adding explicit argument templates to the system prompt; not pooled with the final CPU smoke."
}
```

## Aggregate

```json
{
  "num_trajectories": 8,
  "normalized_accuracy": 0.0,
  "average_cost": 0.0,
  "average_tool_calls": 0.0,
  "deep_search_usage_rate": 0.0,
  "direct_answer_without_tool_rate": 0.0,
  "invalid_action_rate": 1.0,
  "episodes_with_invalid_action_rate": 1.0,
  "truncation_rate": 0.0,
  "tool_execution_errors": 0,
  "termination_counts": {
    "invalid_action": 8
  }
}
```

## Task type

| Type | N | Accuracy | Average cost |
|---|---:|---:|---:|
| list/filter | 2 | 0.000 | 0.000 |
| multi-condition | 2 | 0.000 | 0.000 |
| single-hop | 2 | 0.000 | 0.000 |
| two-hop | 2 | 0.000 | 0.000 |

## Tool / action usage

action_usage 包括终止 action；tool_call_usage 仅以已执行工具调用为分母。

proposed_action_counts 包含可解析出 action 名称但参数验证失败的尝试；unparsed 表示不能可靠提取名称。不要将零次实际调用误解为没有使用工具的意图。

```json
{
  "action_usage": {
    "lookup": {
      "count": 0,
      "fraction": 0
    },
    "search": {
      "count": 0,
      "fraction": 0
    },
    "deep_search": {
      "count": 0,
      "fraction": 0
    },
    "direct_answer": {
      "count": 0,
      "fraction": 0
    }
  },
  "tool_call_usage": {
    "lookup": {
      "count": 0,
      "fraction": 0
    },
    "search": {
      "count": 0,
      "fraction": 0
    },
    "deep_search": {
      "count": 0,
      "fraction": 0
    }
  },
  "first_action_counts": {
    "no_executed_action": 8
  },
  "proposed_action_counts": {
    "deep_search": 15,
    "unparsed": 1
  },
  "first_proposed_action_counts": {
    "deep_search": 7,
    "unparsed": 1
  }
}
```

## Group differences

```json
{
  "task-032": {
    "group_size": 2,
    "unique_tool_sequences": 1,
    "unique_final_answers": 0,
    "missing_final_answers": 2,
    "cost_variance": 0,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
      }
    ]
  },
  "task-033": {
    "group_size": 2,
    "unique_tool_sequences": 1,
    "unique_final_answers": 0,
    "missing_final_answers": 2,
    "cost_variance": 0,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
      }
    ]
  },
  "task-000": {
    "group_size": 2,
    "unique_tool_sequences": 1,
    "unique_final_answers": 0,
    "missing_final_answers": 2,
    "cost_variance": 0,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
      }
    ]
  },
  "task-001": {
    "group_size": 2,
    "unique_tool_sequences": 1,
    "unique_final_answers": 0,
    "missing_final_answers": 2,
    "cost_variance": 0,
    "rollouts": [
      {
        "rollout_id": 0,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
      },
      {
        "rollout_id": 1,
        "tool_sequence": [],
        "final_answer": null,
        "cost": 0,
        "termination_reason": "invalid_action"
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
    "rollout_id": 0,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 2,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  },
  {
    "task_id": "task-032",
    "task_type": "list/filter",
    "rollout_id": 1,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 1,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  },
  {
    "task_id": "task-033",
    "task_type": "multi-condition",
    "rollout_id": 0,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 2,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  },
  {
    "task_id": "task-033",
    "task_type": "multi-condition",
    "rollout_id": 1,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 2,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  },
  {
    "task_id": "task-000",
    "task_type": "single-hop",
    "rollout_id": 0,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 2,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  },
  {
    "task_id": "task-000",
    "task_type": "single-hop",
    "rollout_id": 1,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 2,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  },
  {
    "task_id": "task-001",
    "task_type": "two-hop",
    "rollout_id": 0,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 2,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  },
  {
    "task_id": "task-001",
    "task_type": "two-hop",
    "rollout_id": 1,
    "correctness": false,
    "total_cost": 0,
    "number_of_tool_calls": 0,
    "deep_search_used": false,
    "direct_answer_without_tool": false,
    "termination_reason": "invalid_action",
    "truncated": false,
    "invalid_action_attempts": 2,
    "action_attempts": 2,
    "invalid_arguments": 2,
    "repair_attempts": 1,
    "tool_execution_errors": 0,
    "tool_sequence": [],
    "final_answer": null
  }
]
```

## Observed interface signals

以下仅标记重复的相同调用和列表样式字符串，不自动修正模型答案或影响 policy。

```json
{
  "episodes_with_identical_repeated_calls": [],
  "list_like_string_answers": [],
  "parse_error_counts": {
    "Missing or extra arguments": 7,
    "Invalid JSON: Expecting value: line 1 column 107 (char 106)": 1,
    "query must be an object with table/where/select/reduce only": 7,
    "Invalid field path": 1
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
      "action": "invalid_action",
      "raw_output_indices": [
        0,
        1
      ],
      "cost": 0
    }
  ],
  "tool_calls": [],
  "observations": [],
  "final_answer": null,
  "total_cost": 0,
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
    "{\"action\": \"deep_search\", \"arguments\": {\"table\": \"students\", \"select\": \"id\", \"where\": [{\"field\": \"year\", \"op\": \"eq\", \"value\": \"1\"}]}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"table\": \"students\", \"select\": \"id\", \"where\": [{\"field\": \"year\", \"op\": \"eq\", \"value\": \"1\"}]}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "table": "students",
          "select": "id",
          "where": [
            {
              "field": "year",
              "op": "eq",
              "value": "1"
            }
          ]
        }
      },
      "error": "Missing or extra arguments",
      "invalid_arguments": true
    },
    {
      "step": 1,
      "retry_count": 1,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "table": "students",
          "select": "id",
          "where": [
            {
              "field": "year",
              "op": "eq",
              "value": "1"
            }
          ]
        }
      },
      "error": "Missing or extra arguments",
      "invalid_arguments": true
    }
  ],
  "termination_reason": "invalid_action",
  "truncated": false,
  "number_of_steps": 1,
  "tool_sequence": [],
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
      "action": "invalid_action",
      "raw_output_indices": [
        0,
        1
      ],
      "cost": 0
    }
  ],
  "tool_calls": [],
  "observations": [],
  "final_answer": null,
  "total_cost": 0,
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
    "{\"action\": \"deep_search\", \"arguments\": {\"reduce\": [], \"table\": \"students\", \"where\": [{\"field\": \"advisor\", \"op\": \"eq\", \"value\": \"p00\"}], \"select\": \"id\"}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": \"SELECT students.id FROM students JOIN advisors ON students.advisor = advisors.id WHERE advisors.name = 'p00';\"}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "reduce": [],
          "table": "students",
          "where": [
            {
              "field": "advisor",
              "op": "eq",
              "value": "p00"
            }
          ],
          "select": "id"
        }
      },
      "error": "Missing or extra arguments",
      "invalid_arguments": true
    },
    {
      "step": 1,
      "retry_count": 1,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": "SELECT students.id FROM students JOIN advisors ON students.advisor = advisors.id WHERE advisors.name = 'p00';"
        }
      },
      "error": "query must be an object with table/where/select/reduce only",
      "invalid_arguments": true
    }
  ],
  "termination_reason": "invalid_action",
  "truncated": false,
  "number_of_steps": 1,
  "tool_sequence": [],
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
      "action": "invalid_action",
      "raw_output_indices": [
        0,
        1
      ],
      "cost": 0
    }
  ],
  "tool_calls": [],
  "observations": [],
  "final_answer": null,
  "total_cost": 0,
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
    "{\"action\": \"deep_search\", \"arguments\": {\"table\": \"students\", \"select\": \"id\", \"reduce\": null, \"where\": [{\"field\": \"advisor\", \"op\": \"eq\", \"value\": \"professors.id\"}]}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"table\": \"students\", \"select\": \"id\", \"reduce\": null, \"where\": [{\"field\": \"advisor\", \"op\": \"eq\", \"value\": \"professors.id\"}]}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "table": "students",
          "select": "id",
          "reduce": null,
          "where": [
            {
              "field": "advisor",
              "op": "eq",
              "value": "professors.id"
            }
          ]
        }
      },
      "error": "Missing or extra arguments",
      "invalid_arguments": true
    },
    {
      "step": 1,
      "retry_count": 1,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "table": "students",
          "select": "id",
          "reduce": null,
          "where": [
            {
              "field": "advisor",
              "op": "eq",
              "value": "professors.id"
            }
          ]
        }
      },
      "error": "Missing or extra arguments",
      "invalid_arguments": true
    }
  ],
  "termination_reason": "invalid_action",
  "truncated": false,
  "number_of_steps": 1,
  "tool_sequence": [],
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
      "action": "invalid_action",
      "raw_output_indices": [
        0,
        1
      ],
      "cost": 0
    }
  ],
  "tool_calls": [],
  "observations": [],
  "final_answer": null,
  "total_cost": 0,
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
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": \"SELECT * FROM professors WHERE students.id = 's00'\"}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"query\": \"SELECT * FROM professors WHERE students.id = 's00'\"}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": "SELECT * FROM professors WHERE students.id = 's00'"
        }
      },
      "error": "query must be an object with table/where/select/reduce only",
      "invalid_arguments": true
    },
    {
      "step": 1,
      "retry_count": 1,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "query": "SELECT * FROM professors WHERE students.id = 's00'"
        }
      },
      "error": "query must be an object with table/where/select/reduce only",
      "invalid_arguments": true
    }
  ],
  "termination_reason": "invalid_action",
  "truncated": false,
  "number_of_steps": 1,
  "tool_sequence": [],
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
      "action": "invalid_action",
      "raw_output_indices": [
        0,
        1
      ],
      "cost": 0
    }
  ],
  "tool_calls": [],
  "observations": [],
  "final_answer": null,
  "total_cost": 0,
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
    "{\"action\": \"deep_search\", \"arguments\": {\"reduce\": \"id\", \"select\": \"id\", \"where\": [{\"field\": \"year\", \"op\": eq, \"value\": \"1\"}]}}",
    "{\"action\": \"deep_search\", \"arguments\": {\"reduce\": \"id\", \"select\": \"id\", \"where\": [{\"field\": \"year\", \"op\": \"eq\", \"value\": \"1\"}]}}"
  ],
  "parse_status": [
    {
      "step": 1,
      "retry_count": 0,
      "success": false,
      "parsed_action": null,
      "error": "Invalid JSON: Expecting value: line 1 column 107 (char 106)",
      "invalid_arguments": false
    },
    {
      "step": 1,
      "retry_count": 1,
      "success": false,
      "parsed_action": {
        "action": "deep_search",
        "arguments": {
          "reduce": "id",
          "select": "id",
          "where": [
            {
              "field": "year",
              "op": "eq",
              "value": "1"
            }
          ]
        }
      },
      "error": "Missing or extra arguments",
      "invalid_arguments": true
    }
  ],
  "termination_reason": "invalid_action",
  "truncated": false,
  "number_of_steps": 1,
  "tool_sequence": [],
  "token_ids": [],
  "action_token_mask": [],
  "logprobs": []
}
```
