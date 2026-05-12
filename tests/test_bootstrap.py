from tool_trace_rag.bootstrap import RuntimeBootstrap
from tool_trace_rag.memory.vector_store import TraceVectorStore
from tool_trace_rag.traces.store import TraceStore


class StubProvider:
    provider_name = "stub"
    model = "stub-model"


class StubProviderFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return StubProvider()


def test_runtime_bootstrap_builds_agent_and_eval_factory(tmp_path):
    provider_factory = StubProviderFactory()
    bootstrap = RuntimeBootstrap(provider_factory=provider_factory)

    agent = bootstrap.build_agent(data_path="data/mock_customer_support.json", max_tool_calls=3)
    eval_factory = bootstrap.build_eval_agent_factory(data_path="data/mock_customer_support.json")
    eval_agent = eval_factory(5)

    assert agent.max_tool_calls == 3
    assert eval_agent.max_tool_calls == 5
    assert provider_factory.calls == 2


def test_runtime_bootstrap_builds_trace_and_vector_store(tmp_path):
    bootstrap = RuntimeBootstrap(provider_factory=StubProviderFactory())

    trace_store = bootstrap.build_trace_store(trace_dir=tmp_path / "traces")
    vector_store = bootstrap.build_vector_store(vector_dir=tmp_path / "vectors", collection_name="test")

    assert isinstance(trace_store, TraceStore)
    assert trace_store.root == tmp_path / "traces"
    assert isinstance(vector_store, TraceVectorStore)
    assert vector_store.collection_name == "test"
