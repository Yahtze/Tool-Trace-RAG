from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ToolCallTrace:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | str | None
    error: str | None
    latency_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRunTrace:
    task_id: str | None
    task: str
    messages: list[dict[str, Any]]
    tool_calls: list[ToolCallTrace]
    final_answer: str
    success: bool | None
    provider: str
    model: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return data
