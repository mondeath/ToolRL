from _common import ROOT, write, json
from src.kb import generate_kb
from src.tasks import generate_tasks, public_task
from src.environment import ToolEnvironment
from src.rewards import MockAIReward, CostAwareReward
from src.trajectories import save_jsonl
from src.evaluation import evaluate

def run():
    config = json.loads((ROOT / "configs/default.json").read_text())
    kb = generate_kb(config["seed"])
    tasks = generate_tasks(kb)
    write("kb.json", kb)
    write("tasks.evaluation.json", tasks)
    write("tasks.public.json", [public_task(t) for t in tasks])
    env = ToolEnvironment(kb, config["top_k"], config["max_tool_calls"])
    trajectories = []
    # Scripted evaluation harness uses reference plans; this is not a deployable policy.
    for task in tasks:
        plan = task["evaluation"]["query_plan"]
        for strategy in ("economical", "redundant_deep", "direct_abstain"):
            env.reset(public_task(task), {"strategy":strategy, "seed":config["seed"], "scripted_reference_plan":True})
            if strategy == "direct_abstain": answer = "I do not know"
            elif strategy == "redundant_deep":
                env.step("deep_search", query=plan)
                answer = env.step("deep_search", query=plan)["answer"]
            elif task["task_type"] == "single-hop":
                answer = env.step("lookup", entity=plan["where"][0]["value"], field=plan["select"])["value"]
            elif task["task_type"] == "two-hop":
                advisor = env.step("lookup", entity=plan["where"][0]["value"], field="advisor")["value"]
                answer = env.step("lookup", entity=advisor, field="department")["value"]
            else:
                observation = env.step("search", query=plan)
                if observation["truncated"]: observation = env.step("deep_search", query=plan)
                answer = observation["answer"]
            env.step("direct_answer", final_answer=answer)
            trajectories.append(env.trajectory)
    reward = CostAwareReward(MockAIReward(), config["cost_lambda"])
    summary = evaluate(tasks, trajectories, reward)
    summary["by_strategy"] = {strategy: {k:v for k,v in evaluate(tasks, [t for t in trajectories if t.metadata["strategy"] == strategy], reward).items() if k != "reward_breakdown"} for strategy in ("economical", "redundant_deep", "direct_abstain")}
    save_jsonl(ROOT / "data/trajectories.jsonl", trajectories)
    write("example_trajectory.json", trajectories[0].to_dict())
    write("evaluation_summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "reward_breakdown"}, indent=2))

if __name__ == "__main__": run()
