from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.eval.schema import EvalMetrics, EvalReport, EvalTask, ExpectedToolCall, TaskScore
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace


class AgentFactory(Protocol):
    def __call__(self, max_tool_calls: int) -> ToolCallingAgent:
        ...


def score_trace(task: EvalTask, trace: AgentRunTrace) -> TaskScore:
    reasons: list[str] = []

    if task.requires_tools and not trace.tool_calls:
        reasons.append("expected at least one tool call")
    if not task.requires_tools and trace.tool_calls:
        reasons.append("expected zero tool calls")
    if not trace.final_answer.strip():
        reasons.append("expected non-empty final answer")
    if trace.error:
        reasons.append(f"trace error: {trace.error}")
    if trace.success is False:
        reasons.append("trace success was false")

    for expected_call in task.expected.tool_calls:
        if not _has_expected_tool_call(trace.tool_calls, expected_call):
            reasons.append(_missing_tool_call_reason(expected_call))

    final_answer_lower = trace.final_answer.lower()
    for expected_text in task.expected.answer_contains:
        if expected_text.lower() not in final_answer_lower:
            reasons.append(f"final answer missing expected text: {expected_text}")

    duplicate_tool_calls = _count_duplicate_tool_calls(trace.tool_calls)
    return TaskScore(
        task=task,
        trace=trace,
        passed=not reasons,
        reasons=reasons,
        duplicate_tool_calls=duplicate_tool_calls,
    )


def evaluate_tasks(
    tasks: list[EvalTask],
    agent_factory: Callable[[int], ToolCallingAgent],
    default_max_tool_calls: int = 8,
) -> EvalReport:
    scores: list[TaskScore] = []
    for task in tasks:
        max_tool_calls = task.max_tool_calls if task.max_tool_calls is not None else default_max_tool_calls
        agent = agent_factory(max_tool_calls)
        trace = agent.run(task.prompt, task_id=task.task_id)
        scores.append(score_trace(task, trace))
    return EvalReport(scores=scores, metrics=summarize_scores(scores))


def summarize_scores(scores: list[TaskScore]) -> EvalMetrics:
    total = len(scores)
    passed = sum(1 for score in scores if score.passed)
    failed = total - passed
    tool_call_counts = [len(score.trace.tool_calls) for score in scores]
    required_scores = [score for score in scores if score.task.requires_tools]
    no_tool_scores = [score for score in scores if not score.task.requires_tools]

    return EvalMetrics(
        total_tasks=total,
        passed_tasks=passed,
        failed_tasks=failed,
        success_rate=_safe_divide(passed, total),
        tool_call_rate=_safe_divide(sum(1 for count in tool_call_counts if count > 0), total),
        avg_tool_calls=_average(tool_call_counts),
        avg_tool_calls_when_tools_required=_average([len(score.trace.tool_calls) for score in required_scores]),
        avg_tool_calls_when_tools_not_required=_average([len(score.trace.tool_calls) for score in no_tool_scores]),
        over_tooling_rate=_safe_divide(sum(1 for score in no_tool_scores if score.trace.tool_calls), len(no_tool_scores)),
        under_tooling_rate=_safe_divide(sum(1 for score in required_scores if not score.trace.tool_calls), len(required_scores)),
        max_tool_call_violations=sum(1 for score in scores if score.trace.error == "max_tool_calls_exceeded"),
        duplicate_tool_call_rate=_safe_divide(sum(1 for score in scores if score.duplicate_tool_calls > 0), total),
        error_rate=_safe_divide(sum(1 for score in scores if score.trace.error or score.trace.success is False), total),
    )


def _has_expected_tool_call(actual_calls: list[ToolCallTrace], expected_call: ExpectedToolCall) -> bool:
    return any(
        actual_call.tool_name == expected_call.tool_name
        and _arguments_include(actual_call.arguments, expected_call.arguments)
        for actual_call in actual_calls
    )


def _arguments_include(actual: dict[str, object], expected: dict[str, object]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _missing_tool_call_reason(expected_call: ExpectedToolCall) -> str:
    arguments = json.dumps(expected_call.arguments, sort_keys=True)
    return f"missing expected tool call: {expected_call.tool_name} {arguments}"


def _count_duplicate_tool_calls(tool_calls: list[ToolCallTrace]) -> int:
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for tool_call in tool_calls:
        key = (tool_call.tool_name, json.dumps(tool_call.arguments, sort_keys=True))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _average(values: list[int]) -> float:
    return _safe_divide(sum(values), len(values))


def _safe_divide(numerator: int | float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
