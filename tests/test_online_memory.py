from __future__ import annotations

from pathlib import Path

from tool_trace_rag.memory.injection import MemoryPromptContext
from tool_trace_rag.memory.online import OnlineMemoryConfig, OnlineMemoryRunner
from tool_trace_rag.memory.vector_store import upsert_trace_file
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace
from tool_trace_rag.traces.store import TraceStore


class FakeEmbeddingProvider:
    dimensions = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(texts[0])), 0.0, 1.0]]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


class FakeVectorSink:
    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.upserts: list[dict] = []

    def has_id(self, document_id: str) -> bool:
        return document_id in self.ids

    def upsert(self, document_id: str, text: str, metadata: dict, embedding: list[float]) -> None:
        self.ids.add(document_id)
        self.upserts.append(
            {
                "document_id": document_id,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
            }
        )


class FakeOnlineVectorStore:
    def __init__(self) -> None:
        self.upserted: list[tuple[str, str, bool]] = []

    def upsert_trace_file(self, trace_path, trace_dir, reindex: bool = False) -> str:
        self.upserted.append((str(trace_path), str(trace_dir), reindex))
        return "doc-online-1"


class FakeAgent:
    def __init__(self, trace: AgentRunTrace) -> None:
        self.trace = trace
        self.tasks: list[str] = []

    def run(self, task: str) -> AgentRunTrace:
        self.tasks.append(task)
        return self.trace


class FakeBootstrap:
    def __init__(self, agent: FakeAgent) -> None:
        self.agent = agent
        self.memory_contexts: list[MemoryPromptContext | None] = []

    def build_memory_context(self, **kwargs) -> MemoryPromptContext:
        return MemoryPromptContext(
            prompt_section="Relevant past tool-use examples:\nnone",
            metadata={
                "enabled": True,
                "top_k": kwargs["top_k"],
                "filter": kwargs["memory_filter"],
                "retrieved_count": 0,
                "injected_count": 0,
                "memories": [],
                "error": None,
            },
        )

    def build_agent(self, *, data_path: str, max_tool_calls: int, memory_context: MemoryPromptContext | None = None):
        self.memory_contexts.append(memory_context)
        return self.agent

    def build_trace_store(self, trace_dir):
        return TraceStore(Path(trace_dir))

    def build_vector_store(self, vector_dir, collection_name):
        return FakeOnlineVectorStore()


class FailingTraceStore:
    def write_trace(self, trace: AgentRunTrace):
        raise OSError("disk full")


class FailingUpsertVectorStore:
    def upsert_trace_file(self, trace_path, trace_dir, reindex: bool = False) -> str:
        raise RuntimeError("vector store unavailable")


def sample_trace(task: str = "Can Maya return headphones?") -> AgentRunTrace:
    return AgentRunTrace(
        task_id="task-online-1",
        task=task,
        messages=[{"role": "user", "content": task}],
        tool_calls=[
            ToolCallTrace(
                call_id="call-1",
                tool_name="find_customer",
                arguments={"query": "Maya Chen"},
                result={"customer_id": "cust_001"},
                error=None,
                latency_ms=1.0,
            )
        ],
        final_answer="Maya can return the headphones.",
        success=True,
        provider="fake",
        model="fake-model",
        trace_id="trace-online-1",
        created_at="2026-05-14T00:00:00+00:00",
    )


def test_upsert_trace_file_embeds_and_upserts_one_persisted_trace(tmp_path: Path):
    store = TraceStore(tmp_path)
    trace_path = store.write_trace(sample_trace())
    sink = FakeVectorSink()

    document_id = upsert_trace_file(
        trace_path=trace_path,
        trace_dir=tmp_path,
        sink=sink,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert document_id == "trace-online-1:" + trace_path.name
    assert len(sink.upserts) == 1
    assert sink.upserts[0]["document_id"] == document_id
    assert sink.upserts[0]["metadata"]["relative_source_path"] == trace_path.name
    assert sink.upserts[0]["metadata"]["trace_id"] == "trace-online-1"
    assert "Can Maya return headphones?" in sink.upserts[0]["text"]
    assert sink.upserts[0]["embedding"] == [float(len(sink.upserts[0]["text"])), 0.0, 1.0]


def test_upsert_trace_file_skips_existing_document_by_default(tmp_path: Path):
    store = TraceStore(tmp_path)
    trace_path = store.write_trace(sample_trace())
    sink = FakeVectorSink()
    existing_id = "trace-online-1:" + trace_path.name
    sink.ids.add(existing_id)

    document_id = upsert_trace_file(
        trace_path=trace_path,
        trace_dir=tmp_path,
        sink=sink,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert document_id == existing_id
    assert sink.upserts == []


def test_upsert_trace_file_reindexes_existing_document_when_requested(tmp_path: Path):
    store = TraceStore(tmp_path)
    trace_path = store.write_trace(sample_trace())
    sink = FakeVectorSink()
    sink.ids.add("trace-online-1:" + trace_path.name)

    upsert_trace_file(
        trace_path=trace_path,
        trace_dir=tmp_path,
        sink=sink,
        embedding_provider=FakeEmbeddingProvider(),
        reindex=True,
    )

    assert len(sink.upserts) == 1


def test_online_memory_runner_retrieves_runs_persists_and_upserts(tmp_path: Path):
    trace = sample_trace("Can Maya return headphones?")
    agent = FakeAgent(trace)
    bootstrap = FakeBootstrap(agent)
    vector_store = FakeOnlineVectorStore()
    bootstrap.build_vector_store = lambda vector_dir, collection_name: vector_store
    runner = OnlineMemoryRunner(bootstrap=bootstrap)

    result = runner.run(
        task="Can Maya return headphones?",
        data_path="data/mock_customer_support.json",
        max_tool_calls=8,
        config=OnlineMemoryConfig(
            enabled=True,
            use_memory=True,
            trace_dir=tmp_path,
            vector_dir=tmp_path / "vectors",
            collection_name="test_collection",
            top_k=3,
            memory_filter="successful_only",
            strict=False,
        ),
    )

    assert agent.tasks == ["Can Maya return headphones?"]
    assert bootstrap.memory_contexts[0] is not None
    assert result.trace.trace_id == "trace-online-1"
    assert result.lifecycle["online_memory_enabled"] is True
    assert result.lifecycle["retrieval"]["injected_count"] == 0
    assert result.lifecycle["persistence"]["persisted"] is True
    assert result.lifecycle["persistence"]["trace_path"].endswith(".json")
    assert result.lifecycle["upsert"]["upserted"] is True
    assert result.lifecycle["upsert"]["document_id"] == "doc-online-1"
    assert len(vector_store.upserted) == 1
    assert Path(result.lifecycle["persistence"]["trace_path"]).exists()


def test_online_memory_runner_can_skip_retrieval_but_still_persist_and_upsert(tmp_path: Path):
    trace = sample_trace("No retrieval needed")
    agent = FakeAgent(trace)
    bootstrap = FakeBootstrap(agent)
    vector_store = FakeOnlineVectorStore()
    bootstrap.build_vector_store = lambda vector_dir, collection_name: vector_store
    runner = OnlineMemoryRunner(bootstrap=bootstrap)

    result = runner.run(
        task="No retrieval needed",
        data_path="data/mock_customer_support.json",
        max_tool_calls=8,
        config=OnlineMemoryConfig(
            enabled=True,
            use_memory=False,
            trace_dir=tmp_path,
            vector_dir=tmp_path / "vectors",
            collection_name="test_collection",
            top_k=3,
            memory_filter="successful_only",
            strict=False,
        ),
    )

    assert bootstrap.memory_contexts == [None]
    assert result.lifecycle["retrieval"] == {
        "enabled": False,
        "retrieved_count": 0,
        "injected_count": 0,
        "memories": [],
        "error": None,
    }
    assert result.lifecycle["persistence"]["persisted"] is True
    assert result.lifecycle["upsert"]["upserted"] is True


def test_online_memory_runner_stops_before_upsert_when_persistence_fails(tmp_path: Path):
    agent = FakeAgent(sample_trace())
    bootstrap = FakeBootstrap(agent)
    bootstrap.build_trace_store = lambda trace_dir: FailingTraceStore()
    vector_store = FakeOnlineVectorStore()
    bootstrap.build_vector_store = lambda vector_dir, collection_name: vector_store
    runner = OnlineMemoryRunner(bootstrap=bootstrap)

    result = runner.run(
        task="Can Maya return headphones?",
        data_path="data/mock_customer_support.json",
        max_tool_calls=8,
        config=OnlineMemoryConfig(enabled=True, trace_dir=tmp_path, vector_dir=tmp_path / "vectors"),
    )

    assert result.lifecycle["persistence"] == {"persisted": False, "trace_path": None, "error": "disk full"}
    assert result.lifecycle["upsert"] == {"upserted": False, "document_id": None, "error": None}
    assert vector_store.upserted == []


def test_online_memory_runner_reports_upsert_failure_and_keeps_persisted_trace(tmp_path: Path):
    agent = FakeAgent(sample_trace())
    bootstrap = FakeBootstrap(agent)
    bootstrap.build_vector_store = lambda vector_dir, collection_name: FailingUpsertVectorStore()
    runner = OnlineMemoryRunner(bootstrap=bootstrap)

    result = runner.run(
        task="Can Maya return headphones?",
        data_path="data/mock_customer_support.json",
        max_tool_calls=8,
        config=OnlineMemoryConfig(enabled=True, trace_dir=tmp_path, vector_dir=tmp_path / "vectors"),
    )

    assert result.lifecycle["persistence"]["persisted"] is True
    assert Path(result.lifecycle["persistence"]["trace_path"]).exists()
    assert result.lifecycle["upsert"] == {
        "upserted": False,
        "document_id": None,
        "error": "vector store unavailable",
    }


def test_online_memory_persisted_trace_can_be_read_back_and_vector_metadata_points_to_it(tmp_path: Path):
    trace = sample_trace("Readback task")
    agent = FakeAgent(trace)
    bootstrap = FakeBootstrap(agent)
    sink = FakeVectorSink()

    class VectorStoreUsingSink:
        def upsert_trace_file(self, trace_path, trace_dir, reindex: bool = False) -> str:
            return upsert_trace_file(
                trace_path=trace_path,
                trace_dir=trace_dir,
                sink=sink,
                embedding_provider=FakeEmbeddingProvider(),
                reindex=reindex,
            )

    bootstrap.build_vector_store = lambda vector_dir, collection_name: VectorStoreUsingSink()
    runner = OnlineMemoryRunner(bootstrap=bootstrap)

    result = runner.run(
        task="Readback task",
        data_path="data/mock_customer_support.json",
        max_tool_calls=8,
        config=OnlineMemoryConfig(enabled=True, use_memory=False, trace_dir=tmp_path, vector_dir=tmp_path / "vectors"),
    )

    trace_path = Path(result.lifecycle["persistence"]["trace_path"])
    loaded = TraceStore(tmp_path).read_trace(trace_path)

    assert loaded.trace_id == "trace-online-1"
    assert sink.upserts[0]["metadata"]["relative_source_path"] == trace_path.name
    assert sink.upserts[0]["metadata"]["source_path"] == str(trace_path)
    assert result.lifecycle["upsert"]["document_id"] == "trace-online-1:" + trace_path.name
