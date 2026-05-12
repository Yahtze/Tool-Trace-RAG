from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace


def test_tool_call_trace_serializes_to_dict():
    trace = ToolCallTrace(
        call_id="call_1",
        tool_name="find_customer",
        arguments={"query": "Maya Chen"},
        result={"status": "found", "customer_id": "cust_001"},
        error=None,
        latency_ms=12.5,
    )

    assert trace.to_dict() == {
        "call_id": "call_1",
        "tool_name": "find_customer",
        "arguments": {"query": "Maya Chen"},
        "result": {"status": "found", "customer_id": "cust_001"},
        "error": None,
        "latency_ms": 12.5,
    }


def test_agent_run_trace_supports_no_tool_run():
    trace = AgentRunTrace(
        task_id=None,
        task="What is the standard return window?",
        messages=[{"role": "user", "content": "What is the standard return window?"}],
        tool_calls=[],
        final_answer="Unopened accessories can be returned within 30 days.",
        success=None,
        provider="fake",
        model="fake-model",
    )

    data = trace.to_dict()

    assert data["tool_calls"] == []
    assert data["error"] is None
    assert data["final_answer"] == "Unopened accessories can be returned within 30 days."
