from __future__ import annotations

import time

from tool_trace_rag.providers.base import ToolCall
from tool_trace_rag.tools.registry import ToolRegistry
from tool_trace_rag.traces.schema import ToolCallTrace


def execute_tool_call(tools: ToolRegistry, tool_call: ToolCall) -> ToolCallTrace:
    started = time.perf_counter()
    result = tools.execute(tool_call.name, tool_call.arguments)
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
