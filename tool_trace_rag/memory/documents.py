from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tool_trace_rag.traces.schema import AgentRunTrace

MAX_COMPACT_JSON_CHARS = 500
MAX_PREVIEW_CHARS = 200


@dataclass(frozen=True, slots=True)
class TraceEmbeddingDocument:
    document_id: str
    text: str
    metadata: dict[str, str | int | float | bool | None]


def format_trace_document(
    trace: AgentRunTrace,
    source_path: str | Path,
    relative_source_path: str | None = None,
) -> TraceEmbeddingDocument:
    source = Path(source_path)
    relative = relative_source_path or source.name
    tool_names = [call.tool_name for call in trace.tool_calls]
    lines = [
        f"Trace ID: {trace.trace_id}",
        f"Task ID: {trace.task_id or ''}",
        f"Task: {trace.task}",
        f"Success: {_format_success(trace.success)}",
        f"Provider: {trace.provider}",
        f"Model: {trace.model}",
        f"Final answer: {trace.final_answer}",
        f"Tools used: {' -> '.join(tool_names) if tool_names else 'none'}",
        "Tool calls:",
    ]
    if trace.tool_calls:
        for index, call in enumerate(trace.tool_calls, start=1):
            result_part = f" error={call.error}" if call.error else f" result={_compact_json(call.result)}"
            lines.append(f"{index}. {call.tool_name} args={_compact_json(call.arguments)}{result_part}")
    else:
        lines.append("none")
    metadata = trace_metadata(trace, source_path=source, relative_source_path=relative)
    return TraceEmbeddingDocument(document_id=f"{trace.trace_id}:{relative}", text="\n".join(lines), metadata=metadata)


def trace_metadata(
    trace: AgentRunTrace,
    source_path: str | Path,
    relative_source_path: str | None = None,
) -> dict[str, str | int | float | bool | None]:
    source = Path(source_path)
    relative = relative_source_path or source.name
    return {
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "source_path": str(source),
        "relative_source_path": relative,
        "task_preview": _preview(trace.task),
        "provider": trace.provider,
        "model": trace.model,
        "success": trace.success,
        "created_at": trace.created_at,
        "schema_version": trace.schema_version,
        "tool_names": ",".join(call.tool_name for call in trace.tool_calls),
        "tool_count": len(trace.tool_calls),
    }


def _compact_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)
    if len(text) <= MAX_COMPACT_JSON_CHARS:
        return text
    return text[: MAX_COMPACT_JSON_CHARS - 1] + "…"


def _preview(value: str) -> str:
    text = " ".join(value.split())
    if len(text) <= MAX_PREVIEW_CHARS:
        return text
    return text[: MAX_PREVIEW_CHARS - 1] + "…"


def _format_success(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"
