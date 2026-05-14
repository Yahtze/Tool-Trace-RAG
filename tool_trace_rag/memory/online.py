from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_trace_rag.config import (
    CUSTOMER_SUPPORT_DATA_PATH,
    MEMORY_FILTER,
    MEMORY_STRICT,
    MEMORY_TOP_K,
    TRACE_DIR,
    VECTOR_COLLECTION_NAME,
    VECTOR_DIR,
)
from tool_trace_rag.traces.schema import AgentRunTrace


@dataclass(frozen=True, slots=True)
class OnlineMemoryConfig:
    enabled: bool = False
    use_memory: bool = True
    trace_dir: str | Path = TRACE_DIR
    vector_dir: str | Path = VECTOR_DIR
    collection_name: str = VECTOR_COLLECTION_NAME
    top_k: int = MEMORY_TOP_K
    memory_filter: str = MEMORY_FILTER
    strict: bool = MEMORY_STRICT
    reindex: bool = False


@dataclass(frozen=True, slots=True)
class OnlineMemoryRunResult:
    trace: AgentRunTrace
    lifecycle: dict[str, Any]


class OnlineMemoryRunner:
    def __init__(self, bootstrap: Any) -> None:
        self.bootstrap = bootstrap

    def run(
        self,
        task: str,
        data_path: str = CUSTOMER_SUPPORT_DATA_PATH,
        max_tool_calls: int = 8,
        config: OnlineMemoryConfig = OnlineMemoryConfig(),
    ) -> OnlineMemoryRunResult:
        if not config.enabled:
            agent = self.bootstrap.build_agent(data_path=data_path, max_tool_calls=max_tool_calls, memory_context=None)
            trace = agent.run(task)
            return OnlineMemoryRunResult(trace=trace, lifecycle=_disabled_lifecycle())

        retrieval_metadata = _retrieval_disabled_metadata()
        memory_context = None
        if config.use_memory:
            memory_context = self.bootstrap.build_memory_context(
                task=task,
                trace_dir=config.trace_dir,
                vector_dir=config.vector_dir,
                collection_name=config.collection_name,
                top_k=config.top_k,
                memory_filter=config.memory_filter,
                strict=config.strict,
            )
            retrieval_metadata = memory_context.metadata

        agent = self.bootstrap.build_agent(data_path=data_path, max_tool_calls=max_tool_calls, memory_context=memory_context)
        trace = agent.run(task)

        persistence: dict[str, Any] = {"persisted": False, "trace_path": None, "error": None}
        upsert: dict[str, Any] = {"upserted": False, "document_id": None, "error": None}

        try:
            trace_store = self.bootstrap.build_trace_store(config.trace_dir)
            trace_path = trace_store.write_trace(trace)
            persistence = {"persisted": True, "trace_path": str(trace_path), "error": None}
        except Exception as exc:
            persistence = {"persisted": False, "trace_path": None, "error": str(exc)}
            return OnlineMemoryRunResult(
                trace=trace,
                lifecycle={
                    "online_memory_enabled": True,
                    "retrieval": retrieval_metadata,
                    "persistence": persistence,
                    "upsert": upsert,
                },
            )

        try:
            vector_store = self.bootstrap.build_vector_store(
                vector_dir=config.vector_dir,
                collection_name=config.collection_name,
            )
            document_id = vector_store.upsert_trace_file(trace_path=trace_path, trace_dir=config.trace_dir, reindex=config.reindex)
            upsert = {"upserted": True, "document_id": document_id, "error": None}
        except Exception as exc:
            upsert = {"upserted": False, "document_id": None, "error": str(exc)}

        return OnlineMemoryRunResult(
            trace=trace,
            lifecycle={
                "online_memory_enabled": True,
                "retrieval": retrieval_metadata,
                "persistence": persistence,
                "upsert": upsert,
            },
        )


def _disabled_lifecycle() -> dict[str, Any]:
    return {
        "online_memory_enabled": False,
        "retrieval": _retrieval_disabled_metadata(),
        "persistence": {"persisted": False, "trace_path": None, "error": None},
        "upsert": {"upserted": False, "document_id": None, "error": None},
    }


def _retrieval_disabled_metadata() -> dict[str, Any]:
    return {"enabled": False, "retrieved_count": 0, "injected_count": 0, "memories": [], "error": None}
