from tool_trace_rag.eval.evaluator import evaluate_tasks, score_trace, summarize_scores
from tool_trace_rag.eval.schema import EvalTask, ExpectedToolCall, TaskExpectations
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace
from tool_trace_rag.traces.store import TraceStore


def make_task(
    requires_tools: bool,
    expected_tool_calls: list[ExpectedToolCall] | None = None,
    answer_contains: list[str] | None = None,
) -> EvalTask:
    return EvalTask(
        task_id="m2_test",
        prompt="Test prompt",
        requires_tools=requires_tools,
        expected=TaskExpectations(
            tool_calls=expected_tool_calls or [],
            answer_contains=answer_contains or [],
        ),
        max_tool_calls=None,
        tags=[],
    )


def make_trace(
    tool_calls: list[ToolCallTrace] | None = None,
    final_answer: str = "Final answer with ord_1001.",
    error: str | None = None,
    success: bool | None = None,
) -> AgentRunTrace:
    return AgentRunTrace(
        task_id="m2_test",
        task="Test prompt",
        messages=[],
        tool_calls=tool_calls or [],
        final_answer=final_answer,
        success=success,
        provider="fake",
        model="fake-model",
        error=error,
    )


def call(tool_name: str, arguments: dict[str, object]) -> ToolCallTrace:
    return ToolCallTrace(
        call_id=f"call_{tool_name}",
        tool_name=tool_name,
        arguments=arguments,
        result={"status": "ok"},
        error=None,
        latency_ms=1.0,
    )


def test_score_tool_required_task_passes_with_expected_tool_and_answer():
    task = make_task(
        requires_tools=True,
        expected_tool_calls=[ExpectedToolCall("check_refund_eligibility", {"order_id": "ord_1001"})],
        answer_contains=["ord_1001"],
    )
    trace = make_trace(tool_calls=[call("check_refund_eligibility", {"order_id": "ord_1001"})])

    score = score_trace(task, trace)

    assert score.passed is True
    assert score.reasons == []


def test_score_tool_required_task_fails_without_tool_call():
    task = make_task(requires_tools=True)
    trace = make_trace(tool_calls=[])

    score = score_trace(task, trace)

    assert score.passed is False
    assert "expected at least one tool call" in score.reasons


def test_score_no_tool_task_passes_with_zero_tools_and_answer():
    task = make_task(requires_tools=False, answer_contains=["30 days"])
    trace = make_trace(tool_calls=[], final_answer="The return window is 30 days.")

    score = score_trace(task, trace)

    assert score.passed is True
    assert score.reasons == []


def test_score_no_tool_task_fails_when_tool_was_used():
    task = make_task(requires_tools=False)
    trace = make_trace(tool_calls=[call("find_customer", {"query": "Maya Chen"})])

    score = score_trace(task, trace)

    assert score.passed is False
    assert "expected zero tool calls" in score.reasons


def test_score_fails_on_missing_answer_substring_case_insensitive():
    task = make_task(requires_tools=False, answer_contains=["30 days"])
    trace = make_trace(tool_calls=[], final_answer="Returns are allowed for one month.")

    score = score_trace(task, trace)

    assert score.passed is False
    assert "final answer missing expected text: 30 days" in score.reasons


def test_score_detects_duplicate_tool_calls():
    task = make_task(requires_tools=True)
    trace = make_trace(
        tool_calls=[
            call("find_customer", {"query": "Maya Chen"}),
            call("find_customer", {"query": "Maya Chen"}),
        ]
    )

    score = score_trace(task, trace)

    assert score.duplicate_tool_calls == 1


def test_summarize_scores_computes_required_metrics():
    tool_task = make_task(requires_tools=True)
    no_tool_task = make_task(requires_tools=False)
    scores = [
        score_trace(tool_task, make_trace(tool_calls=[call("find_customer", {"query": "Maya Chen"})])),
        score_trace(tool_task, make_trace(tool_calls=[], error="max_tool_calls_exceeded", success=False)),
        score_trace(no_tool_task, make_trace(tool_calls=[])),
        score_trace(no_tool_task, make_trace(tool_calls=[call("find_customer", {"query": "Maya Chen"})])),
    ]

    metrics = summarize_scores(scores)

    assert metrics.total_tasks == 4
    assert metrics.passed_tasks == 2
    assert metrics.failed_tasks == 2
    assert metrics.success_rate == 0.5
    assert metrics.tool_call_rate == 0.5
    assert metrics.avg_tool_calls == 0.5
    assert metrics.avg_tool_calls_when_tools_required == 0.5
    assert metrics.avg_tool_calls_when_tools_not_required == 0.5
    assert metrics.over_tooling_rate == 0.5
    assert metrics.under_tooling_rate == 0.5
    assert metrics.max_tool_call_violations == 1
    assert metrics.error_rate == 0.25


class StubAgent:
    def __init__(self, traces):
        self._traces = list(traces)
        self.calls = []

    def run(self, task: str, task_id: str | None = None):
        self.calls.append((task, task_id))
        return self._traces.pop(0)


def test_evaluate_tasks_uses_task_specific_max_tool_call_override():
    task = EvalTask(
        task_id="m2_override",
        prompt="Find order ord_1001.",
        requires_tools=True,
        expected=TaskExpectations(tool_calls=[], answer_contains=[]),
        max_tool_calls=2,
        tags=[],
    )
    created_max_values = []

    def factory(max_tool_calls: int):
        created_max_values.append(max_tool_calls)
        return StubAgent([make_trace(tool_calls=[call("get_order", {"order_id": "ord_1001"})])])

    report = evaluate_tasks([task], factory, default_max_tool_calls=8)

    assert created_max_values == [2]
    assert report.metrics.total_tasks == 1
    assert report.scores[0].trace.task_id == "m2_test"


def test_evaluate_tasks_can_persist_traces_without_changing_scores(tmp_path):
    task = make_task(requires_tools=False, answer_contains=["30 days"])
    trace = make_trace(tool_calls=[], final_answer="The return window is 30 days.")
    store = TraceStore(tmp_path)

    def factory(max_tool_calls: int):
        return StubAgent([trace])

    report = evaluate_tasks([task], factory, default_max_tool_calls=8, trace_store=store)

    assert report.metrics.total_tasks == 1
    assert report.metrics.passed_tasks == 1
    persisted = store.list_traces()
    assert len(persisted) == 1
    assert persisted[0].to_dict() == trace.to_dict()
