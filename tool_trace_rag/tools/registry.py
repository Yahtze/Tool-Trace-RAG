from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., dict[str, Any] | str]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | str:
        tool = self._tools.get(tool_name)
        if tool is None:
            return {
                "status": "tool_error",
                "error_code": "UNKNOWN_TOOL",
                "message": f"Tool '{tool_name}' is not registered.",
            }

        try:
            return tool.function(**arguments)
        except Exception as exc:
            return {
                "status": "tool_error",
                "error_code": "EXCEPTION",
                "message": str(exc),
            }
