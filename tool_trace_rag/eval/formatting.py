from __future__ import annotations

from tool_trace_rag.eval.schema import EvalReport

_PERCENT_FIELDS = {
    "success_rate",
    "tool_call_rate",
    "over_tooling_rate",
    "under_tooling_rate",
    "duplicate_tool_call_rate",
    "error_rate",
}


def format_eval_report(report: EvalReport) -> str:
    metrics = report.metrics
    lines = [
        "Evaluation summary",
        f"Total tasks: {metrics.total_tasks}",
        f"Passed: {metrics.passed_tasks}",
        f"Failed: {metrics.failed_tasks}",
        "",
        "Metrics:",
    ]

    for name in (
        "success_rate",
        "tool_call_rate",
        "avg_tool_calls",
        "avg_tool_calls_when_tools_required",
        "avg_tool_calls_when_tools_not_required",
        "over_tooling_rate",
        "under_tooling_rate",
        "max_tool_call_violations",
        "duplicate_tool_call_rate",
        "error_rate",
    ):
        value = getattr(metrics, name)
        lines.append(f"{name}: {_format_metric(name, value)}")

    failures = [score for score in report.scores if not score.passed]
    if failures:
        lines.extend(["", "Failures:"])
        for score in failures:
            reason = "; ".join(score.reasons)
            lines.append(f"{score.task.task_id}: {reason}")
    else:
        lines.extend(["", "Failures: none"])

    return "\n".join(lines)


def _format_metric(name: str, value: float | int) -> str:
    if name in _PERCENT_FIELDS:
        return f"{float(value) * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)
