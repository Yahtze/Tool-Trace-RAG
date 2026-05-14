from __future__ import annotations

from argparse import Namespace

import scripts.run_task as run_task
from tool_trace_rag.memory.online import OnlineMemoryRunResult
from tool_trace_rag.traces.schema import AgentRunTrace
from tool_trace_rag.traces.store import TraceStore


def test_run_task_online_memory_output(monkeypatch, capsys, tmp_path):
    trace = AgentRunTrace(
        task_id="task-cli",
        task="CLI task",
        messages=[{"role": "user", "content": "CLI task"}],
        tool_calls=[],
        final_answer="CLI answer",
        success=True,
        provider="fake",
        model="fake-model",
        trace_id="trace-cli",
        created_at="2026-05-14T00:00:00+00:00",
    )
    result = OnlineMemoryRunResult(
        trace=trace,
        lifecycle={
            "online_memory_enabled": True,
            "retrieval": {"enabled": True, "retrieved_count": 0, "injected_count": 0, "memories": [], "error": None},
            "persistence": {"persisted": True, "trace_path": str(tmp_path / "trace.json"), "error": None},
            "upsert": {"upserted": True, "document_id": "trace-cli:trace.json", "error": None},
        },
    )

    class FakeRunner:
        def __init__(self, bootstrap):
            self.bootstrap = bootstrap

        def run(self, **kwargs):
            return result

    monkeypatch.setattr(run_task, "OnlineMemoryRunner", FakeRunner)
    args = Namespace(
        task="CLI task",
        data="data/mock_customer_support.json",
        max_tool_calls=8,
        save_trace=False,
        trace_dir=str(tmp_path),
        use_memory=True,
        online_memory=True,
        vector_dir=str(tmp_path / "vectors"),
        collection="test_collection",
        top_k=3,
        memory_filter="successful_only",
        memory_strict=False,
    )

    run_task.run_with_args(args, bootstrap=object())

    output = capsys.readouterr().out
    assert "Online memory: enabled" in output
    assert "Retrieved memories: 0" in output
    assert "Trace persisted: " in output
    assert "Vector upserted: true" in output
    assert "Vector document: trace-cli:trace.json" in output
    assert "Final answer:" in output
    assert "CLI answer" in output


def test_run_task_use_memory_without_online_memory_does_not_use_online_runner(monkeypatch, capsys, tmp_path):
    called = {"online": False}

    class FailingOnlineRunner:
        def __init__(self, bootstrap):
            called["online"] = True

    class FakeAgentForCli:
        def run(self, task: str) -> AgentRunTrace:
            return AgentRunTrace(
                task_id="task-retrieval-only",
                task=task,
                messages=[{"role": "user", "content": task}],
                tool_calls=[],
                final_answer="retrieval-only answer",
                success=True,
                provider="fake",
                model="fake-model",
                retrieval={"enabled": True, "injected_count": 0, "memories": [], "error": None},
            )

    class FakeBootstrapForCli:
        def build_memory_context(self, **kwargs):
            return object()

        def build_agent(self, **kwargs):
            return FakeAgentForCli()

        def build_trace_store(self, trace_dir):
            return TraceStore(trace_dir)

    monkeypatch.setattr(run_task, "OnlineMemoryRunner", FailingOnlineRunner)
    args = Namespace(
        task="Retrieval only task",
        data="data/mock_customer_support.json",
        max_tool_calls=8,
        save_trace=False,
        trace_dir=None,
        use_memory=True,
        online_memory=False,
        vector_dir=str(tmp_path / "vectors"),
        collection="test_collection",
        top_k=3,
        memory_filter="successful_only",
        memory_strict=False,
    )

    run_task.run_with_args(args, bootstrap=FakeBootstrapForCli())

    output = capsys.readouterr().out
    assert called["online"] is False
    assert "Online memory: enabled" not in output
    assert "Vector upserted" not in output
    assert "retrieval-only answer" in output
