from __future__ import annotations

import json

from tool_trace_rag.agent_loop import AgentLoop
from tool_trace_rag.providers.base import AssistantMessage, ToolCall
from tool_trace_rag.tools.registry import ToolDefinition, ToolRegistry


class RecordingProvider:
    provider_name = "recording"
    model = "recording-model"

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    def complete(self, messages, tools, tool_choice="auto"):
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("No responses left")
        return self._responses.pop(0)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            function=lambda value: {"status": "ok", "value": value},
        )
    )
    return registry


def test_agent_loop_appends_tool_result_before_next_provider_turn():
    provider = RecordingProvider(
        [
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_1", name="echo", arguments={"value": "hello"})]),
            AssistantMessage(content="done", tool_calls=[]),
        ]
    )

    loop = AgentLoop(provider=provider, tools=_registry(), system_prompt="sys", policy_context="policy", max_tool_calls=3)
    trace = loop.run("task")

    assert trace.final_answer == "done"
    assert len(provider.calls) == 2
    second_turn_messages = provider.calls[1]
    assert second_turn_messages[-1]["role"] == "tool"
    assert second_turn_messages[-1]["tool_call_id"] == "call_1"
    assert json.loads(second_turn_messages[-1]["content"]) == {"status": "ok", "value": "hello"}


def test_execute_tool_error_is_reflected_in_trace_error_field():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="broken",
            description="Broken",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    )
    provider = RecordingProvider(
        [
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_1", name="broken", arguments={})]),
            AssistantMessage(content="failed", tool_calls=[]),
        ]
    )

    loop = AgentLoop(provider=provider, tools=registry, system_prompt="sys", policy_context="policy", max_tool_calls=3)
    trace = loop.run("task")

    assert trace.tool_calls[0].error == "EXCEPTION"
    assert trace.tool_calls[0].result["status"] == "tool_error"
