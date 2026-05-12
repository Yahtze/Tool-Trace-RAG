from __future__ import annotations

from typing import Any

from tool_trace_rag.providers.base import AssistantMessage


class FakeProvider:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AssistantMessage:
        if not self._responses:
            raise RuntimeError("FakeProvider has no scripted responses left.")
        self.call_count += 1
        return self._responses.pop(0)
