from pathlib import Path

import pytest

from tool_trace_rag.memory.retrieval import MemoryRetrievalConfig, MemoryRetriever, MemoryRetrievalError
from tool_trace_rag.memory.vector_store import QueryResult
from tool_trace_rag.traces.schema import AgentRunTrace
from tool_trace_rag.traces.store import TraceStore


def make_trace(trace_id: str, success: bool | None = True) -> AgentRunTrace:
    return AgentRunTrace(
        trace_id=trace_id,
        created_at="2026-05-12T00:00:00+00:00",
        task_id=f"task-{trace_id}",
        task=f"Task {trace_id}",
        messages=[],
        tool_calls=[],
        final_answer="Done",
        success=success,
        provider="fake",
        model="fake-model",
    )


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def query(self, task: str, top_k: int):
        return self.results[:top_k]


class FailingVectorStore:
    def query(self, task: str, top_k: int):
        raise RuntimeError("store unavailable")


def test_retriever_resolves_vector_results_to_traces(tmp_path):
    store = TraceStore(tmp_path)
    path = store.write_trace(make_trace("trace-1"))
    vector = FakeVectorStore([QueryResult(1, f"trace-1:{path.name}", 0.2, "doc", {"relative_source_path": path.name, "trace_id": "trace-1"})])

    result = MemoryRetriever(vector, tmp_path).retrieve("new task", MemoryRetrievalConfig(top_k=3))

    assert result.error is None
    assert len(result.examples) == 1
    assert result.examples[0].trace.trace_id == "trace-1"
    assert result.metadata["memories"][0]["source_path"] == path.name


def test_retriever_filters_successful_only(tmp_path):
    store = TraceStore(tmp_path)
    ok = store.write_trace(make_trace("ok", success=True))
    bad = store.write_trace(make_trace("bad", success=False))
    vector = FakeVectorStore([
        QueryResult(1, f"bad:{bad.name}", 0.1, "doc", {"relative_source_path": bad.name, "trace_id": "bad"}),
        QueryResult(2, f"ok:{ok.name}", 0.2, "doc", {"relative_source_path": ok.name, "trace_id": "ok"}),
    ])

    result = MemoryRetriever(vector, tmp_path).retrieve("new task", MemoryRetrievalConfig(top_k=3, filter="successful_only"))

    assert [example.trace.trace_id for example in result.examples] == ["ok"]


def test_retriever_no_results_returns_empty_metadata(tmp_path):
    result = MemoryRetriever(FakeVectorStore([]), tmp_path).retrieve("new task", MemoryRetrievalConfig(top_k=3))

    assert result.examples == []
    assert result.metadata["retrieved_count"] == 0
    assert result.metadata["injected_count"] == 0


def test_retriever_non_strict_failure_returns_metadata(tmp_path):
    result = MemoryRetriever(FailingVectorStore(), tmp_path).retrieve("new task", MemoryRetrievalConfig(strict=False))

    assert result.examples == []
    assert "store unavailable" in result.metadata["error"]


def test_retriever_strict_failure_raises_controlled_error(tmp_path):
    with pytest.raises(MemoryRetrievalError):
        MemoryRetriever(FailingVectorStore(), tmp_path).retrieve("new task", MemoryRetrievalConfig(strict=True))


def test_memory_package_exports_retrieval_apis():
    from tool_trace_rag.memory import MemoryPromptContext, MemoryRetrievalConfig, MemoryRetriever, format_memory_prompt_section

    assert MemoryPromptContext is not None
    assert MemoryRetrievalConfig is not None
    assert MemoryRetriever is not None
    assert format_memory_prompt_section is not None
