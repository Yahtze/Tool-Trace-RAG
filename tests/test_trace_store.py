from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace
from tool_trace_rag.traces.store import TraceStore


def make_trace(trace_id: str = "trace-fixed", success: bool | None = True, error: str | None = None) -> AgentRunTrace:
    return AgentRunTrace(
        trace_id=trace_id,
        created_at="2026-05-12T00:00:00+00:00",
        task_id="m3_trace",
        task="Persist this trace",
        messages=[{"role": "user", "content": "Persist this trace"}],
        tool_calls=[
            ToolCallTrace("call_1", "find_customer", {"query": "Maya"}, {"status": "ok"}, None, 1.0),
            ToolCallTrace("call_2", "get_order", {"order_id": "ord_1001"}, {"status": "ok"}, None, 2.0),
        ],
        final_answer="Done",
        success=success,
        provider="fake",
        model="fake-model",
        error=error,
    )


def test_write_and_read_trace_round_trips_required_fields(tmp_path):
    store = TraceStore(tmp_path)
    trace = make_trace()

    path = store.write_trace(trace)
    loaded = store.read_trace(path)

    assert path.parent == tmp_path
    assert path.suffix == ".json"
    assert loaded.to_dict() == trace.to_dict()
    assert [call.call_id for call in loaded.tool_calls] == ["call_1", "call_2"]


def test_failed_trace_persists_error_metadata(tmp_path):
    store = TraceStore(tmp_path)
    trace = make_trace(trace_id="trace-failed", success=False, error="max_tool_calls_exceeded")

    loaded = store.read_trace(store.write_trace(trace))

    assert loaded.success is False
    assert loaded.error == "max_tool_calls_exceeded"


def test_multiple_traces_do_not_overwrite(tmp_path):
    store = TraceStore(tmp_path)
    first = make_trace(trace_id="same-id")
    second = make_trace(trace_id="same-id")

    first_path = store.write_trace(first)
    second_path = store.write_trace(second)

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()


def test_list_traces_returns_records_sorted_by_created_at_then_path(tmp_path):
    store = TraceStore(tmp_path)
    later = make_trace(trace_id="later")
    later.created_at = "2026-05-12T00:01:00+00:00"
    earlier = make_trace(trace_id="earlier")
    earlier.created_at = "2026-05-12T00:00:00+00:00"

    store.write_trace(later)
    store.write_trace(earlier)

    traces = store.list_traces()

    assert [trace.trace_id for trace in traces] == ["earlier", "later"]
