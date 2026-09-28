"""Offline evaluation only. List answers use unordered set semantics."""
import json
import unicodedata

def normalize(value):
    if isinstance(value, str): return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if isinstance(value, list): return sorted(set(json.dumps(normalize(v), sort_keys=True) for v in value))
    if isinstance(value, dict): return {k: normalize(v) for k, v in sorted(value.items())}
    return value

def normalized_equal(actual, expected): return normalize(actual) == normalize(expected)

def evaluate(tasks, trajectories, reward):
    if not trajectories: raise ValueError("No trajectories")
    indexed = {task["task_id"]: task for task in tasks}
    breakdown = []
    for t in trajectories:
        if not t.steps or t.steps[-1]["action"] != "direct_answer": raise ValueError("Unfinished trajectory")
        truth = indexed[t.task_id]["ground_truth"]
        breakdown.append(dict(task_id=t.task_id, strategy=t.metadata.get("strategy"),
                              exact_accuracy=float(t.final_answer == truth),
                              normalized_accuracy=float(normalized_equal(t.final_answer, truth)),
                              tool_calls=len(t.tool_calls),
                              deep_search_calls=sum(c["tool_name"] == "deep_search" for c in t.tool_calls),
                              **reward.breakdown(t)))
    n = len(breakdown)
    avg = lambda key: sum(r[key] for r in breakdown)/n
    calls = sum(r["tool_calls"] for r in breakdown)
    return {"num_trajectories":n, "exact_accuracy":avg("exact_accuracy"),
            "normalized_accuracy":avg("normalized_accuracy"), "average_tool_cost":avg("total_cost"),
            "average_tool_calls":avg("tool_calls"),
            "deep_search_usage_rate":sum(r["deep_search_calls"] > 0 for r in breakdown)/n,
            "deep_search_call_fraction":sum(r["deep_search_calls"] for r in breakdown)/calls if calls else 0,
            "average_ai_reward":avg("ai_reward"), "average_pure_reward":avg("pure_reward"),
            "average_cost_penalty":avg("cost_penalty"), "average_cost_aware_reward":avg("cost_aware_reward"),
            "reward_breakdown":breakdown}
