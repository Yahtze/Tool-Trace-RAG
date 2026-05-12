from dataclasses import dataclass

from tool_trace_rag.memory.ingestion import IndexSummary, TraceIngestionModule


@dataclass
class Entry:
    name: str


@dataclass
class Record:
    document_id: str
    text: str
    metadata: dict[str, str]


class FakeSource:
    def __init__(self) -> None:
        self._entries = [Entry("ok-1"), Entry("bad"), Entry("ok-2")]

    def list_entries(self):
        return self._entries

    def load_entry(self, entry: Entry) -> Record:
        if entry.name == "bad":
            raise ValueError("broken")
        return Record(document_id=entry.name, text=f"text-{entry.name}", metadata={"entry": entry.name})

    def entry_label(self, entry: Entry) -> str:
        return entry.name


class FakeEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


class FakeSink:
    def __init__(self) -> None:
        self.ids: set[str] = set()

    def has_id(self, document_id: str) -> bool:
        return document_id in self.ids

    def upsert(self, document_id: str, text: str, metadata: dict[str, str], embedding: list[float]) -> None:
        self.ids.add(document_id)


def test_trace_ingestion_module_indexes_skips_and_collects_failures():
    module = TraceIngestionModule(embedding_provider=FakeEmbeddingProvider())
    source = FakeSource()
    sink = FakeSink()

    first = module.index(source=source, sink=sink)
    second = module.index(source=source, sink=sink)

    assert first == IndexSummary(indexed_traces=2, skipped_duplicates=0, failed_traces=1, errors=["bad: broken"])
    assert second.indexed_traces == 0
    assert second.skipped_duplicates == 2
    assert second.failed_traces == 1
