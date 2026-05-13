from __future__ import annotations

from tool_trace_rag.agent_loop import AgentLoop
from tool_trace_rag.config import AGENT_MAX_TOOL_CALLS
from tool_trace_rag.memory.injection import MemoryPromptContext
from tool_trace_rag.providers.base import ChatProvider
from tool_trace_rag.tools.registry import ToolRegistry
from tool_trace_rag.traces.schema import AgentRunTrace

DEFAULT_SYSTEM_PROMPT = """You are a customer support assistant. Use tools only when needed to retrieve customer, order, or refund information. If the answer is general policy knowledge present in the prompt, answer directly without tools."""

POLICY_CONTEXT = """General policy: unopened accessories, apparel, and home items can be returned within 30 days after delivery. Opened electronics are not refundable. Final sale clearance items are not refundable. Orders that have not been delivered yet are not eligible for refund."""


class ToolCallingAgent:
    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_calls: int = AGENT_MAX_TOOL_CALLS,
        memory_context: MemoryPromptContext | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_tool_calls = max_tool_calls
        self.memory_context = memory_context

    def run(self, task: str, task_id: str | None = None) -> AgentRunTrace:
        loop = AgentLoop(
            provider=self.provider,
            tools=self.tools,
            system_prompt=self.system_prompt,
            policy_context=POLICY_CONTEXT,
            max_tool_calls=self.max_tool_calls,
            memory_prompt_section=self.memory_context.prompt_section if self.memory_context else "",
            retrieval_metadata=self.memory_context.metadata if self.memory_context else None,
        )
        return loop.run(task=task, task_id=task_id)

