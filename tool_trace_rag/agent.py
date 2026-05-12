from __future__ import annotations

import json
import time
from typing import Any

from tool_trace_rag.config import AGENT_MAX_TOOL_CALLS
from tool_trace_rag.providers.base import ChatProvider, ToolCall
from tool_trace_rag.tools.registry import ToolRegistry
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace

DEFAULT_SYSTEM_PROMPT = """You are a customer support assistant. Use tools only when needed to retrieve customer, order, or refund information. If the answer is general policy knowledge present in the prompt, answer directly without tools."""

POLICY_CONTEXT = """General policy: unopened accessories, apparel, and home items can be returned within 30 days after delivery. Opened electronics are not refundable. Final sale clearance items are not refundable. Orders that have not been delivered yet are not eligible for refund."""


class ToolCallingAgent:
    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_calls: int = AGENT_MAX_TOOL_CALLS,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_tool_calls = max_tool_calls

    def run(self, task: str, task_id: str | None = None) -> AgentRunTrace:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{POLICY_CONTEXT}"},
            {"role": "user", "content": task},
        ]
        trace_calls: list[ToolCallTrace] = []

        while True:
            assistant = self.provider.complete(
                messages=messages,
                tools=self.tools.tool_schemas(),
                tool_choice="auto",
            )
            assistant_message = self._assistant_message_to_dict(assistant.content, assistant.tool_calls)
            messages.append(assistant_message)

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
                    )
                trace_call = self._execute_tool_call(tool_call)
                trace_calls.append(trace_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(trace_call.result),
                    }
                )

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolCallTrace:
        started = time.perf_counter()
        result = self.tools.execute(tool_call.name, tool_call.arguments)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        error = None
        if isinstance(result, dict) and result.get("status") == "tool_error":
            error = str(result.get("error_code", "TOOL_ERROR"))
        return ToolCallTrace(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _assistant_message_to_dict(content: str | None, tool_calls: list[ToolCall]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in tool_calls
            ]
        return message
