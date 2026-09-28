"""Minimal policy boundary. Inference dependencies are loaded only by HuggingFacePolicy."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class AgentState:
    question: str
    history: list
    action_schema: dict


@dataclass
class PolicyOutput:
    raw_output: str
    token_ids: list = field(default_factory=list)


class Policy(Protocol):
    model_name: str
    sampling_config: dict

    def new_session(self, seed: int) -> 'Policy':
        """Return an isolated sampling session; immutable model weights may be shared."""
        ...

    def act(self, state: AgentState) -> PolicyOutput: ...
