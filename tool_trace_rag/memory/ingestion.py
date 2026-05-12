from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from tool_trace_rag.memory.documents import TraceEmbeddingDocument, format_trace_document
from tool_trace_rag.memory.embeddings import EmbeddingProvider
from tool_trace_rag.traces.store import TraceStore


@dataclass(frozen=True, slots=True)
class IndexSummary:
    indexed_traces: int = 0
    skipped_duplicates: int = 0
    failed_traces: int = 0
    errors: list[str] = field(default_factory=list)


TEntry = TypeVar("TEntry")


class TraceRecordSource(Protocol[TEntry]):
    def list_entries(self) -> list[TEntry]: ...

    def load_entry(self, entry: TEntry) -> TraceEmbeddingDocument: ...

    def entry_label(self, entry: TEntry) -> str: ...


class VectorDocumentSink(Protocol):
    def has_id(self, document_id: str) -> bool: ...

    def upsert(self, document_id: str, text: str, metadata: dict[str, Any], embedding: list[float]) -> None: ...


class TraceIngestionModule:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding_provider = embedding_provider

    def index(self, source: TraceRecordSource[TEntry], sink: VectorDocumentSink, reindex: bool = False) -> IndexSummary:
        indexed = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        for entry in source.list_entries():
            try:
                document = source.load_entry(entry)
                if not reindex and sink.has_id(document.document_id):
                    skipped += 1
                    continue
                embedding = self._embedding_provider.embed_documents([document.text])[0]
                sink.upsert(document.document_id, document.text, document.metadata, embedding)
                indexed += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{source.entry_label(entry)}: {exc}")

        return IndexSummary(indexed_traces=indexed, skipped_duplicates=skipped, failed_traces=failed, errors=errors)


class FileSystemTraceSource:
    def __init__(self, trace_dir: str | Path) -> None:
        self._root = Path(trace_dir)
        self._store = TraceStore(self._root)

    def list_entries(self) -> list[Path]:
        if not self._root.exists():
            return []
        return sorted(self._root.glob("*.json"))

    def load_entry(self, entry: Path) -> TraceEmbeddingDocument:
        relative = entry.relative_to(self._root).as_posix()
        trace = self._store.read_trace(entry)
        return format_trace_document(trace, source_path=entry, relative_source_path=relative)

    def entry_label(self, entry: Path) -> str:
        return entry.name
