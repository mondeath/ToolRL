import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

@dataclass
class Trajectory:
    task_id: str
    question: str
    steps: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    final_answer: Any = None
    total_cost: float = 0
    metadata: dict = field(default_factory=dict)

    policy_model: Any = None
    sampling_config: dict = field(default_factory=dict)
    raw_outputs: list = field(default_factory=list)
    parse_status: list = field(default_factory=list)
    termination_reason: Any = None
    truncated: bool = False
    number_of_steps: int = 0
    tool_sequence: list = field(default_factory=list)
    token_ids: list = field(default_factory=list)
    action_token_mask: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)

    def to_dict(self): return asdict(self)

def save_jsonl(path, trajectories):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        for trajectory in trajectories:
            row = trajectory.to_dict() if isinstance(trajectory, Trajectory) else trajectory
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
