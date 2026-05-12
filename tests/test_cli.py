import subprocess
import sys

from tool_trace_rag.cli import format_trace_summary
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace


def test_format_trace_summary_includes_final_answer_and_tool_calls():
    trace = AgentRunTrace(
        task_id=None,
        task="Find Maya",
        messages=[],
        tool_calls=[
            ToolCallTrace(
                call_id="call_1",
                tool_name="find_customer",
                arguments={"query": "Maya Chen"},
                result={"status": "found", "customer": {"customer_id": "cust_001"}},
                error=None,
                latency_ms=1.2,
            )
        ],
        final_answer="Maya was found.",
        success=None,
        provider="fake",
        model="fake-model",
    )

    output = format_trace_summary(trace)

    assert "Final answer:\nMaya was found." in output
    assert "Tool calls: 1" in output
    assert "1. find_customer {\"query\": \"Maya Chen\"} -> found" in output


def test_format_trace_summary_handles_no_tool_calls():
    trace = AgentRunTrace(
        task_id=None,
        task="Policy question",
        messages=[],
        tool_calls=[],
        final_answer="30 days.",
        success=None,
        provider="fake",
        model="fake-model",
    )

    output = format_trace_summary(trace)

    assert "Final answer:\n30 days." in output
    assert "Tool calls: 0" in output


def test_run_eval_help_documents_trace_flags():
    result = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--save-traces" in result.stdout
    assert "--trace-dir" in result.stdout


def test_run_task_help_documents_trace_flags():
    result = subprocess.run(
        [sys.executable, "scripts/run_task.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--save-trace" in result.stdout
    assert "--trace-dir" in result.stdout


def test_index_traces_help_documents_vector_flags():
    result = subprocess.run(
        [sys.executable, "scripts/index_traces.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--trace-dir" in result.stdout
    assert "--vector-dir" in result.stdout
    assert "--collection" in result.stdout
    assert "--reindex" in result.stdout


def test_query_traces_help_documents_query_flags():
    result = subprocess.run(
        [sys.executable, "scripts/query_traces.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "task" in result.stdout
    assert "--top-k" in result.stdout
    assert "--vector-dir" in result.stdout
    assert "--collection" in result.stdout
