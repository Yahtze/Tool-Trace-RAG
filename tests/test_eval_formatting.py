from tool_trace_rag.eval.formatting import format_eval_report
from tool_trace_rag.eval.schema import EvalMetrics, EvalReport
from tests.test_evaluator import make_task, make_trace
from tool_trace_rag.eval.evaluator import score_trace


def test_format_eval_report_includes_metrics_and_failures():
    passing_score = score_trace(make_task(requires_tools=False), make_trace(tool_calls=[]))
    failing_score = score_trace(make_task(requires_tools=False), make_trace(tool_calls=[], final_answer=""))
    metrics = EvalMetrics(
        total_tasks=2,
        passed_tasks=1,
        failed_tasks=1,
        success_rate=0.5,
        tool_call_rate=0.0,
        avg_tool_calls=0.0,
        avg_tool_calls_when_tools_required=0.0,
        avg_tool_calls_when_tools_not_required=0.0,
        over_tooling_rate=0.0,
        under_tooling_rate=0.0,
        max_tool_call_violations=0,
        duplicate_tool_call_rate=0.0,
        error_rate=0.0,
    )
    report = EvalReport(scores=[passing_score, failing_score], metrics=metrics)

    output = format_eval_report(report)

    assert "Total tasks: 2" in output
    assert "Passed: 1" in output
    assert "success_rate: 50.00%" in output
    assert "Failures:" in output
    assert "m2_test: expected non-empty final answer" in output
