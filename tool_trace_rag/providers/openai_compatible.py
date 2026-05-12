from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from tool_trace_rag.config import AGENT_API_KEY, AGENT_BASE_URL, AGENT_MODEL, AGENT_TIMEOUT_SECONDS
from tool_trace_rag.providers.base import AssistantMessage, ToolCall


class OpenAICompatibleProvider:
    provider_name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = AGENT_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        base_url = AGENT_BASE_URL
        api_key = AGENT_API_KEY
        model = AGENT_MODEL
        if not api_key:
            raise RuntimeError("AGENT_API_KEY is required.")
        if not model:
            raise RuntimeError("AGENT_MODEL is required.")
        return cls(base_url=base_url, api_key=api_key, model=model)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AssistantMessage:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Provider HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Provider request failed: {exc.reason}") from exc
        return self._parse_response(response_payload)

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> AssistantMessage:
        message = payload["choices"][0]["message"]
        tool_calls = []
        for raw_call in message.get("tool_calls") or []:
            call_id = raw_call["id"]
            function = raw_call["function"]
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool call '{call_id}' arguments are not valid JSON.") from exc
            tool_calls.append(ToolCall(id=call_id, name=function["name"], arguments=arguments))
        return AssistantMessage(content=message.get("content"), tool_calls=tool_calls)

