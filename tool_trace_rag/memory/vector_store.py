from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from tool_trace_rag.config import QUERY_TOP_K, VECTOR_COLLECTION_NAME, VECTOR_DIR
from tool_trace_rag.memory.documents import format_trace_document
from tool_trace_rag.memory.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from tool_trace_rag.traces.store import TraceStore

DEFAULT_VECTOR_DIR = VECTOR_DIR
DEFAULT_COLLECTION_NAME = VECTOR_COLLECTION_NAME


@dataclass(frozen=True, slots=True)
class IndexSummary:
    indexed_traces: int = 0
    skipped_duplicates: int = 0
    failed_traces: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QueryResult:
    rank: int
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class TraceVectorStore:
    def __init__(
        self,
        vector_dir: str | Path = DEFAULT_VECTOR_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.vector_dir = Path(vector_dir)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider()
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.vector_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def index_directory(self, trace_dir: str | Path, reindex: bool = False) -> IndexSummary:
        root = Path(trace_dir)
        if not root.exists():
            return IndexSummary()
        indexed = 0
        skipped = 0
        failed = 0
        errors: list[str] = []
        trace_store = TraceStore(root)
        for path in sorted(root.glob("*.json")):
            try:
                relative = path.relative_to(root).as_posix()
                trace = trace_store.read_trace(path)
                document = format_trace_document(trace, source_path=path, relative_source_path=relative)
                if not reindex and self._has_id(document.document_id):
                    skipped += 1
                    continue
                embedding = self.embedding_provider.embed_documents([document.text])[0]
                self._collection.upsert(
                    ids=[document.document_id],
                    documents=[document.text],
                    metadatas=[document.metadata],
                    embeddings=[embedding],
                )
                indexed += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{path.name}: {exc}")
        return IndexSummary(indexed_traces=indexed, skipped_duplicates=skipped, failed_traces=failed, errors=errors)

    def query(self, task: str, top_k: int = QUERY_TOP_K) -> list[QueryResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        embedding = self.embedding_provider.embed_query(task)
        raw = self._collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        results: list[QueryResult] = []
        for index, document_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 0.0
            results.append(
                QueryResult(
                    rank=index + 1,
                    document_id=str(document_id),
                    score=distance,
                    text=str(documents[index]) if index < len(documents) else "",
                    metadata=dict(metadatas[index]) if index < len(metadatas) and metadatas[index] is not None else {},
                )
            )
        return results

    def _has_id(self, document_id: str) -> bool:
        existing = self._collection.get(ids=[document_id], include=[])
        return bool(existing.get("ids"))
