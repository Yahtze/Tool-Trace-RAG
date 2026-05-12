from __future__ import annotations

import json

from tool_trace_rag.traces.schema import AgentRunTrace


def format_trace_summary(trace: AgentRunTrace) -> str:
    lines = ["Final answer:", trace.final_answer, "", f"Tool calls: {len(trace.tool_calls)}"]
    for index, call in enumerate(trace.tool_calls, start=1):
        args = json.dumps(call.arguments, sort_keys=True)
        result_status = _result_status(call.result)
        lines.append(f"{index}. {call.tool_name} {args} -> {result_status}")
    if trace.error:
        lines.extend(["", f"Run error: {trace.error}"])
    return "\n".join(lines)


def _result_status(result: object) -> str:
    if isinstance(result, dict):
        if "status" in result:
            return str(result["status"])
        if "eligible" in result:
            return "eligible" if result["eligible"] else "not eligible"
    return "ok"
