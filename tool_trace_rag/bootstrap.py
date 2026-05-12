from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.memory.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_DIR,
    TraceVectorStore,
)
from tool_trace_rag.providers.openai_compatible import OpenAICompatibleProvider
from tool_trace_rag.tools.customer_support import build_customer_support_registry
from tool_trace_rag.traces.store import DEFAULT_TRACE_DIR, TraceStore


class RuntimeBootstrap:
    def __init__(self, provider_factory: Callable[[], object] | None = None) -> None:
        self._provider_factory = provider_factory or OpenAICompatibleProvider.from_env

    def build_agent(self, data_path: str, max_tool_calls: int) -> ToolCallingAgent:
        provider = self._provider_factory()
        tools = build_customer_support_registry(data_path)
        return ToolCallingAgent(provider=provider, tools=tools, max_tool_calls=max_tool_calls)

    def build_eval_agent_factory(self, data_path: str) -> Callable[[int], ToolCallingAgent]:
        def _factory(max_tool_calls: int) -> ToolCallingAgent:
            return self.build_agent(data_path=data_path, max_tool_calls=max_tool_calls)

        return _factory

    def build_trace_store(self, trace_dir: str | Path = DEFAULT_TRACE_DIR) -> TraceStore:
        return TraceStore(trace_dir)

    def build_vector_store(
        self,
        vector_dir: str | Path = DEFAULT_VECTOR_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> TraceVectorStore:
        return TraceVectorStore(vector_dir=vector_dir, collection_name=collection_name)
