from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

TRACE_SCHEMA_VERSION = "agent-run-trace/v1"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallTrace":
        return cls(
            call_id=str(data["call_id"]),
            tool_name=str(data["tool_name"]),
            arguments=dict(data.get("arguments", {})),
            result=data.get("result"),
            error=data.get("error"),
            latency_ms=data.get("latency_ms"),
        )


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
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRunTrace":
        return cls(
            task_id=data.get("task_id"),
            task=str(data["task"]),
            messages=list(data.get("messages", [])),
            tool_calls=[ToolCallTrace.from_dict(item) for item in data.get("tool_calls", [])],
            final_answer=str(data.get("final_answer", "")),
            success=data.get("success"),
            provider=str(data["provider"]),
            model=str(data["model"]),
            error=data.get("error"),
            trace_id=str(data["trace_id"]),
            created_at=str(data["created_at"]),
            schema_version=str(data.get("schema_version", TRACE_SCHEMA_VERSION)),
        )
