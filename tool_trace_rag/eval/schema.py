from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tool_trace_rag.traces.schema import AgentRunTrace


@dataclass(frozen=True, slots=True)
class ExpectedToolCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TaskExpectations:
    tool_calls: list[ExpectedToolCall]
    answer_contains: list[str]


@dataclass(frozen=True, slots=True)
class EvalTask:
    task_id: str
    prompt: str
    requires_tools: bool
    expected: TaskExpectations
    max_tool_calls: int | None
    tags: list[str]


@dataclass(frozen=True, slots=True)
class TaskScore:
    task: EvalTask
    trace: AgentRunTrace
    passed: bool
    reasons: list[str]
    duplicate_tool_calls: int


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    success_rate: float
    tool_call_rate: float
    avg_tool_calls: float
    avg_tool_calls_when_tools_required: float
    avg_tool_calls_when_tools_not_required: float
    over_tooling_rate: float
    under_tooling_rate: float
    max_tool_call_violations: int
    duplicate_tool_call_rate: float
    error_rate: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    scores: list[TaskScore]
    metrics: EvalMetrics
