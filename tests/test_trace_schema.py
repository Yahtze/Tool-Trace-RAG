from tool_trace_rag.traces.schema import TRACE_SCHEMA_VERSION, AgentRunTrace, ToolCallTrace


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


def test_agent_run_trace_defaults_persistence_metadata():
    trace = AgentRunTrace(
        task_id="m3_001",
        task="Test task",
        messages=[{"role": "user", "content": "Test task"}],
        tool_calls=[],
        final_answer="Done",
        success=True,
        provider="fake",
        model="fake-model",
    )

    data = trace.to_dict()

    assert data["schema_version"] == TRACE_SCHEMA_VERSION
    assert data["trace_id"]
    assert data["created_at"].endswith("+00:00")


def test_agent_run_trace_round_trips_retrieval_metadata():
    trace = AgentRunTrace(
        trace_id="trace-current",
        created_at="2026-05-12T00:00:00+00:00",
        task_id="m5_current",
        task="Current task",
        messages=[{"role": "user", "content": "Current task"}],
        tool_calls=[],
        final_answer="Done",
        success=True,
        provider="fake",
        model="fake-model",
        retrieval={
            "enabled": True,
            "top_k": 3,
            "filter": "successful_only",
            "retrieved_count": 1,
            "injected_count": 1,
            "memories": [
                {"rank": 1, "trace_id": "trace-old", "task_id": "old-task", "score": 0.25, "source_path": "old.json"}
            ],
            "error": None,
        },
    )

    loaded = AgentRunTrace.from_dict(trace.to_dict())

    assert loaded.retrieval == trace.retrieval
    assert loaded.to_dict()["retrieval"]["memories"][0]["trace_id"] == "trace-old"


def test_agent_run_trace_from_dict_preserves_tool_call_order_and_metadata():
    data = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": "trace-fixed",
        "created_at": "2026-05-12T00:00:00+00:00",
        "task_id": "m3_002",
        "task": "Use tools",
        "messages": [{"role": "user", "content": "Use tools"}],
        "tool_calls": [
            {"call_id": "call_1", "tool_name": "find_customer", "arguments": {"query": "Maya"}, "result": {"status": "ok"}, "error": None, "latency_ms": 1.0},
            {"call_id": "call_2", "tool_name": "get_order", "arguments": {"order_id": "ord_1001"}, "result": {"status": "ok"}, "error": None, "latency_ms": 2.0},
        ],
        "final_answer": "Done",
        "success": True,
        "provider": "fake",
        "model": "fake-model",
        "error": None,
    }

    trace = AgentRunTrace.from_dict(data)

    assert trace.trace_id == "trace-fixed"
    assert trace.created_at == "2026-05-12T00:00:00+00:00"
    assert [call.call_id for call in trace.tool_calls] == ["call_1", "call_2"]
    assert trace.to_dict() == {**data, "retrieval": None}
