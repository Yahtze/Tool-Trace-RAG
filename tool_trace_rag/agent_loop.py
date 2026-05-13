from __future__ import annotations

import json
from typing import Any

from tool_trace_rag.providers.base import AssistantMessage, ChatProvider, ToolCall
from tool_trace_rag.tool_execution import execute_tool_call
from tool_trace_rag.tools.registry import ToolRegistry
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace


class AgentLoop:
    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        system_prompt: str,
        policy_context: str,
        max_tool_calls: int,
        memory_prompt_section: str = "",
        retrieval_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.policy_context = policy_context
        self.max_tool_calls = max_tool_calls
        self.memory_prompt_section = memory_prompt_section
        self.retrieval_metadata = retrieval_metadata

    def run(self, task: str, task_id: str | None = None) -> AgentRunTrace:
        system_parts = [self.system_prompt, self.policy_context]
        if self.memory_prompt_section.strip():
            system_parts.append(self.memory_prompt_section.strip())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": task},
        ]
        trace_calls: list[ToolCallTrace] = []

        while True:
            assistant = self.provider.complete(
                messages=messages,
                tools=self.tools.tool_schemas(),
                tool_choice="auto",
            )
            messages.append(self.assistant_message_to_dict(assistant))

            if not assistant.tool_calls:
                return AgentRunTrace(
                    task_id=task_id,
                    task=task,
                    messages=messages,
                    tool_calls=trace_calls,
                    final_answer=assistant.content or "",
                    success=None,
                    provider=self.provider.provider_name,
                    model=self.provider.model,
                    retrieval=self.retrieval_metadata,
                )

            for tool_call in assistant.tool_calls:
                if len(trace_calls) >= self.max_tool_calls:
                    return AgentRunTrace(
                        task_id=task_id,
                        task=task,
                        messages=messages,
                        tool_calls=trace_calls,
                        final_answer="",
                        success=False,
                        provider=self.provider.provider_name,
                        model=self.provider.model,
                        error="max_tool_calls_exceeded",
                        retrieval=self.retrieval_metadata,
                    )
                trace_call = execute_tool_call(self.tools, tool_call)
                trace_calls.append(trace_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(trace_call.result),
                    }
                )

    @staticmethod
    def assistant_message_to_dict(assistant: AssistantMessage) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": assistant.content}
        if assistant.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in assistant.tool_calls
            ]
        return message
