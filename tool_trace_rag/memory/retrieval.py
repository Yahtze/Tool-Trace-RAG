from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tool_trace_rag.config import MEMORY_FILTER, MEMORY_STRICT, MEMORY_TOP_K
from tool_trace_rag.memory.formatting import MemoryExample
from tool_trace_rag.memory.vector_store import QueryResult
from tool_trace_rag.traces.store import TraceStore


class VectorSearch(Protocol):
    def query(self, task: str, top_k: int) -> list[QueryResult]: ...


class MemoryRetrievalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MemoryRetrievalConfig:
    top_k: int = MEMORY_TOP_K
    filter: str = MEMORY_FILTER
    strict: bool = MEMORY_STRICT


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    examples: list[MemoryExample]
    metadata: dict[str, Any]
    error: str | None = None


class MemoryRetriever:
    def __init__(self, vector_store: VectorSearch, trace_dir: str | Path) -> None:
        self.vector_store = vector_store
        self.trace_dir = Path(trace_dir)
        self.trace_store = TraceStore(self.trace_dir)

    def retrieve(self, task: str, config: MemoryRetrievalConfig) -> MemoryRetrievalResult:
        try:
            raw_results = self.vector_store.query(task, top_k=config.top_k)
            examples: list[MemoryExample] = []
            memories: list[dict[str, Any]] = []
            for result in sorted(raw_results, key=lambda item: (item.rank, item.document_id)):
                path_text = str(result.metadata.get("relative_source_path") or result.metadata.get("source_path") or "")
                if not path_text:
                    continue
                path = self.trace_dir / path_text if not Path(path_text).is_absolute() else Path(path_text)
                trace = self.trace_store.read_trace(path)
                if not _passes_filter(trace.success, config.filter):
                    continue
                examples.append(MemoryExample(rank=len(examples) + 1, score=result.score, trace=trace, source_path=path_text))
                memories.append(
                    {
                        "rank": result.rank,
                        "trace_id": trace.trace_id,
                        "task_id": trace.task_id,
                        "score": result.score,
                        "source_path": path_text,
                        "success": trace.success,
                    }
                )
            metadata = {
                "enabled": True,
                "top_k": config.top_k,
                "filter": config.filter,
                "retrieved_count": len(raw_results),
                "injected_count": len(examples),
                "memories": memories,
                "error": None,
            }
            return MemoryRetrievalResult(examples=examples, metadata=metadata)
        except Exception as exc:
            if config.strict:
                raise MemoryRetrievalError(str(exc)) from exc
            metadata = {
                "enabled": True,
                "top_k": config.top_k,
                "filter": config.filter,
                "retrieved_count": 0,
                "injected_count": 0,
                "memories": [],
                "error": str(exc),
            }
            return MemoryRetrievalResult(examples=[], metadata=metadata, error=str(exc))


def _passes_filter(success: bool | None, policy: str) -> bool:
    if policy == "all":
        return True
    if policy == "successful_only":
        return success is True
    if policy == "failed_only":
        return success is False
    raise ValueError("memory filter must be one of: successful_only, failed_only, all")
