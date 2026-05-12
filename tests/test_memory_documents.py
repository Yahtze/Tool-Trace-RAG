from pathlib import Path

from tool_trace_rag.memory.documents import format_trace_document, trace_metadata
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace


def make_trace() -> AgentRunTrace:
    return AgentRunTrace(
        trace_id="trace-001",
        created_at="2026-05-12T00:00:00+00:00",
        task_id="refund-headphones",
        task="Customer asks whether delivered headphones are refundable.",
        messages=[{"role": "user", "content": "Customer asks whether delivered headphones are refundable."}],
        tool_calls=[
            ToolCallTrace(
                call_id="call_1",
                tool_name="find_customer",
                arguments={"query": "Maya Chen"},
                result={"status": "found", "customer": {"customer_id": "cust_001", "name": "Maya Chen"}},
                error=None,
                latency_ms=1.0,
            ),
            ToolCallTrace(
                call_id="call_2",
                tool_name="get_order",
                arguments={"order_id": "ord_1001"},
                result={"status": "found", "order": {"order_id": "ord_1001", "status": "delivered"}},
                error=None,
                latency_ms=2.0,
            ),
        ],
        final_answer="The headphones are refundable within 30 days of delivery.",
        success=True,
        provider="fake",
        model="fake-model",
    )


def test_format_trace_document_is_deterministic_and_observable_only():
    trace = make_trace()

    document = format_trace_document(trace, source_path=Path("runs/traces/trace.json"), relative_source_path="trace.json")

    assert document.document_id == "trace-001:trace.json"
    assert document.text == """Trace ID: trace-001
Task ID: refund-headphones
Task: Customer asks whether delivered headphones are refundable.
Success: true
Provider: fake
Model: fake-model
Final answer: The headphones are refundable within 30 days of delivery.
Tools used: find_customer -> get_order
Tool calls:
1. find_customer args={\"query\": \"Maya Chen\"} result={\"customer\": {\"customer_id\": \"cust_001\", \"name\": \"Maya Chen\"}, \"status\": \"found\"}
2. get_order args={\"order_id\": \"ord_1001\"} result={\"order\": {\"order_id\": \"ord_1001\", \"status\": \"delivered\"}, \"status\": \"found\"}"""
    assert "messages" not in document.text.lower()


def test_trace_metadata_is_compact_json_serializable():
    metadata = trace_metadata(make_trace(), source_path=Path("/tmp/traces/trace.json"), relative_source_path="trace.json")

    assert metadata == {
        "trace_id": "trace-001",
        "task_id": "refund-headphones",
        "source_path": "/tmp/traces/trace.json",
        "relative_source_path": "trace.json",
        "task_preview": "Customer asks whether delivered headphones are refundable.",
        "provider": "fake",
        "model": "fake-model",
        "success": True,
        "created_at": "2026-05-12T00:00:00+00:00",
        "schema_version": "agent-run-trace/v1",
        "tool_names": "find_customer,get_order",
        "tool_count": 2,
    }
