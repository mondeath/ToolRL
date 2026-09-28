"""One episode per task; direct_answer is the terminal, zero-cost action."""
from copy import deepcopy
from src.tools import QueryEngine
from src.trajectories import Trajectory

class ToolEnvironment:
    COSTS = {"lookup": 1, "search": 2, "deep_search": 8}

    def __init__(self, kb, top_k=5, max_tool_calls=12):
        if top_k < 1 or max_tool_calls < 1: raise ValueError("Limits must be positive")
        self.engine = QueryEngine(deepcopy(kb))
        self.top_k, self.max_tool_calls = top_k, max_tool_calls
        self.trajectory = None
        self.done = False

    def reset(self, task, metadata=None):
        if set(task) != {"task_id", "question"}: raise ValueError("Pass public_task only; evaluation data is forbidden")
        self.trajectory = Trajectory(**task, metadata=deepcopy(metadata or {}))
        self.done = False
        return deepcopy(task)

    def step(self, action, **arguments):
        if self.trajectory is None or self.done: raise RuntimeError("Reset required or episode already finished")
        t = self.trajectory
        if action == "direct_answer":
            if set(arguments) != {"final_answer"}: raise ValueError("direct_answer requires final_answer")
            t.final_answer = deepcopy(arguments["final_answer"])
            t.steps.append({"action": action, "arguments": deepcopy(arguments), "cost": 0})
            self.done = True
            return {"done": True}
        if action not in self.COSTS: raise ValueError("Unknown action")
        if len(t.tool_calls) >= self.max_tool_calls: raise RuntimeError("Tool budget exhausted; submit direct_answer")
        try:
            if action == "lookup": observation = self.engine.lookup(**arguments)
            elif action == "search": observation = self.engine.search(top_k=self.top_k, **arguments)
            else: observation = self.engine.execute(**arguments)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            observation = {"error": str(exc), "error_type": type(exc).__name__}
        # Malformed calls to known tools still incur cost.
        call = {"tool_name": action, "arguments": deepcopy(arguments), "observation": deepcopy(observation), "cost": self.COSTS[action]}
        t.tool_calls.append(call)
        t.observations.append(deepcopy(observation))
        t.steps.append({"action": action, "tool_call_index": len(t.tool_calls)-1, "cost": call["cost"]})
        t.total_cost += call["cost"]
        return deepcopy(observation)
