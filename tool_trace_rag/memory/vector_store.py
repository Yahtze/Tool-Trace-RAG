from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from tool_trace_rag.config import QUERY_TOP_K, VECTOR_COLLECTION_NAME, VECTOR_DIR
from tool_trace_rag.memory.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from tool_trace_rag.memory.ingestion import FileSystemTraceSource, IndexSummary, TraceIngestionModule

DEFAULT_VECTOR_DIR = VECTOR_DIR
DEFAULT_COLLECTION_NAME = VECTOR_COLLECTION_NAME


@dataclass(frozen=True, slots=True)
class QueryResult:
    rank: int
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class _ChromaVectorDocumentSink:
    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def has_id(self, document_id: str) -> bool:
        existing = self._collection.get(ids=[document_id], include=[])
        return bool(existing.get("ids"))

    def upsert(self, document_id: str, text: str, metadata: dict[str, Any], embedding: list[float]) -> None:
        self._collection.upsert(
            ids=[document_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding],
        )


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
        self._ingestion = TraceIngestionModule(embedding_provider=self.embedding_provider)

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def index_directory(self, trace_dir: str | Path, reindex: bool = False) -> IndexSummary:
        source = FileSystemTraceSource(trace_dir)
        sink = _ChromaVectorDocumentSink(self._collection)
        return self._ingestion.index(source=source, sink=sink, reindex=reindex)

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
