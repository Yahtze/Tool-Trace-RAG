from tool_trace_rag.memory.embeddings import EmbeddingProvider


class FakeEmbeddingProvider:
    dimensions = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count(" ")), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count(" ")), 1.0]


def test_fake_embedding_provider_matches_protocol():
    provider: EmbeddingProvider = FakeEmbeddingProvider()

    assert provider.embed_documents(["alpha", "alpha beta"]) == [[5.0, 0.0, 1.0], [10.0, 1.0, 1.0]]
    assert provider.embed_query("alpha") == [5.0, 0.0, 1.0]


from tool_trace_rag.memory.vector_store import TraceVectorStore
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace
from tool_trace_rag.traces.store import TraceStore


def make_trace(trace_id: str, task: str, tool_name: str = "find_customer") -> AgentRunTrace:
    return AgentRunTrace(
        trace_id=trace_id,
        created_at="2026-05-12T00:00:00+00:00",
        task_id=f"task-{trace_id}",
        task=task,
        messages=[{"role": "user", "content": task}],
        tool_calls=[ToolCallTrace("call_1", tool_name, {"query": "Maya"}, {"status": "ok"}, None, 1.0)],
        final_answer="Done",
        success=True,
        provider="fake",
        model="fake-model",
    )


def test_index_directory_indexes_traces_and_skips_duplicates(tmp_path):
    trace_dir = tmp_path / "traces"
    store = TraceStore(trace_dir)
    store.write_trace(make_trace("trace-1", "Find Maya order"))
    store.write_trace(make_trace("trace-2", "Explain return window", "lookup_policy"))
    vector_store = TraceVectorStore(tmp_path / "vectors", embedding_provider=FakeEmbeddingProvider())

    first = vector_store.index_directory(trace_dir)
    second = vector_store.index_directory(trace_dir)

    assert first.indexed_traces == 2
    assert first.skipped_duplicates == 0
    assert first.failed_traces == 0
    assert second.indexed_traces == 0
    assert second.skipped_duplicates == 2
    assert vector_store.count() == 2


def test_query_returns_ranked_structured_results_with_metadata(tmp_path):
    trace_dir = tmp_path / "traces"
    store = TraceStore(trace_dir)
    path = store.write_trace(make_trace("trace-1", "Find Maya order"))
    vector_store = TraceVectorStore(tmp_path / "vectors", embedding_provider=FakeEmbeddingProvider())
    vector_store.index_directory(trace_dir)

    results = vector_store.query("Find a customer order", top_k=1)

    assert len(results) == 1
    result = results[0]
    assert result.rank == 1
    assert isinstance(result.score, float)
    assert result.document_id == f"trace-1:{path.name}"
    assert result.metadata["trace_id"] == "trace-1"
    assert result.metadata["relative_source_path"] == path.name
    assert "Task: Find Maya order" in result.text


def test_empty_trace_directory_returns_zero_summary(tmp_path):
    vector_store = TraceVectorStore(tmp_path / "vectors", embedding_provider=FakeEmbeddingProvider())

    summary = vector_store.index_directory(tmp_path / "empty")

    assert summary.indexed_traces == 0
    assert summary.skipped_duplicates == 0
    assert summary.failed_traces == 0
    assert summary.errors == []


def test_malformed_trace_file_fails_gracefully(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    bad = trace_dir / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    vector_store = TraceVectorStore(tmp_path / "vectors", embedding_provider=FakeEmbeddingProvider())

    summary = vector_store.index_directory(trace_dir)

    assert summary.indexed_traces == 0
    assert summary.failed_traces == 1
    assert "bad.json" in summary.errors[0]


def test_rebuild_clears_existing_collection(tmp_path):
    trace_dir = tmp_path / "traces"
    store = TraceStore(trace_dir)
    store.write_trace(make_trace("trace-1", "Find Maya order"))
    vector_store = TraceVectorStore(tmp_path / "vectors", embedding_provider=FakeEmbeddingProvider())
    vector_store.index_directory(trace_dir)

    vector_store.clear()

    assert vector_store.count() == 0


def test_memory_package_exports_public_apis():
    from tool_trace_rag.memory import (
        DEFAULT_COLLECTION_NAME,
        DEFAULT_VECTOR_DIR,
        IndexSummary,
        QueryResult,
        TraceEmbeddingDocument,
        TraceVectorStore,
        format_trace_document,
        trace_metadata,
    )

    assert DEFAULT_COLLECTION_NAME == "tool_trace_memory"
    assert str(DEFAULT_VECTOR_DIR) == "runs/vector_store"
    assert IndexSummary is not None
    assert QueryResult is not None
    assert TraceEmbeddingDocument is not None
    assert TraceVectorStore is not None
    assert format_trace_document is not None
    assert trace_metadata is not None
