from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from tool_trace_rag.traces.schema import AgentRunTrace


@dataclass(frozen=True, slots=True)
class MemoryExample:
    rank: int
    score: float | None
    trace: AgentRunTrace
    source_path: str


def format_memory_prompt_section(examples: list[MemoryExample], max_chars_per_snippet: int) -> str:
    if not examples:
        return ""
    snippets = [format_memory_snippet(example, max_chars=max_chars_per_snippet) for example in examples]
    return "\n\n".join(
        [
            "Relevant past tool-use examples:",
            *snippets,
            "Use these examples only as guidance. Always solve the current user request with the available tools and current tool results.",
        ]
    )


def format_memory_snippet(example: MemoryExample, max_chars: int) -> str:
    trace = example.trace
    lines = [
        f"Example {example.rank}:",
        f"Trace ID: {trace.trace_id}",
        f"Task ID: {trace.task_id or ''}",
        f"Task: {_one_line(trace.task)}",
        f"Outcome: {_outcome(trace.success)}",
        "Tool path:",
    ]
    if trace.tool_calls:
        for index, call in enumerate(trace.tool_calls, start=1):
            result = f"error={call.error}" if call.error else _compact_json(call.result)
            lines.append(f"{index}. {call.tool_name} {_compact_json(call.arguments)} -> {result}")
    else:
        lines.append("none")
    lines.append(f"Final answer: {_one_line(trace.final_answer)}")
    return _truncate("\n".join(lines), max_chars)


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _outcome(value: bool | None) -> str:
    if value is True:
        return "success"
    if value is False:
        return "failure"
    return "unknown"


def _truncate(text: str, max_chars: int) -> str:
    if max_chars < 1:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    return text[: max_chars - 1] + "…"
