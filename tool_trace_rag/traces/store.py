from __future__ import annotations

import json
import re
from pathlib import Path

from tool_trace_rag.config import TRACE_DIR
from tool_trace_rag.traces.schema import AgentRunTrace

DEFAULT_TRACE_DIR = TRACE_DIR


class TraceStore:
    def __init__(self, root: str | Path = DEFAULT_TRACE_DIR) -> None:
        self.root = Path(root)

    def write_trace(self, trace: AgentRunTrace) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._unique_path(trace)
        path.write_text(json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def read_trace(self, path: str | Path) -> AgentRunTrace:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return AgentRunTrace.from_dict(data)

    def list_traces(self) -> list[AgentRunTrace]:
        if not self.root.exists():
            return []
        traces = [self.read_trace(path) for path in self.root.glob("*.json")]
        return sorted(traces, key=lambda trace: (trace.created_at, trace.trace_id))

    def _unique_path(self, trace: AgentRunTrace) -> Path:
        base = self._base_filename(trace)
        candidate = self.root / f"{base}.json"
        suffix = 2
        while candidate.exists():
            candidate = self.root / f"{base}-{suffix}.json"
            suffix += 1
        return candidate

    def _base_filename(self, trace: AgentRunTrace) -> str:
        created = re.sub(r"[^0-9A-Za-z]+", "", trace.created_at)[:20] or "trace"
        task = _slug(trace.task_id or trace.task)[:48]
        return f"{created}-{task}-{_slug(trace.trace_id)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip()).strip("-._")
    return slug or "trace"
