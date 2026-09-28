"""Quality scorer never receives costs, tool names, or ground truth."""
import math
from typing import Protocol, Any

class AIReward(Protocol):
    def score(self, question: str, final_answer: Any, evidence: list) -> float: ...

class MockAIReward:
    """Evidence-agreement stub, NOT an AI judge or correctness oracle."""
    def score(self, question, final_answer, evidence):
        from src.evaluation import normalized_equal
        return float(any(normalized_equal(final_answer, item[key])
                         for item in evidence for key in ("answer", "value") if key in item))

class CostReward:
    def score(self, tool_calls): return sum(call["cost"] for call in tool_calls)

class PureReward:
    def __init__(self, ai_reward): self.ai_reward = ai_reward
    def score(self, trajectory):
        return self.ai_reward.score(trajectory.question, trajectory.final_answer, trajectory.observations)

class CostAwareReward(PureReward):
    def __init__(self, ai_reward, cost_lambda=0.1):
        super().__init__(ai_reward)
        if not math.isfinite(cost_lambda) or cost_lambda < 0: raise ValueError("lambda must be finite and nonnegative")
        self.cost_lambda = cost_lambda
    def breakdown(self, trajectory):
        quality = super().score(trajectory)
        cost = CostReward().score(trajectory.tool_calls)
        if cost != trajectory.total_cost: raise ValueError("Trajectory cost mismatch")
        penalty = self.cost_lambda * cost
        return {"ai_reward": quality, "total_cost": cost, "cost_penalty": penalty, "pure_reward": quality, "cost_aware_reward": quality-penalty}
    def score(self, trajectory): return self.breakdown(trajectory)["cost_aware_reward"]
