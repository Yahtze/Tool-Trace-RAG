from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tool_trace_rag.eval.schema import EvalTask, ExpectedToolCall, TaskExpectations


def load_eval_tasks(path: str | Path) -> list[EvalTask]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON array.")

    tasks: list[EvalTask] = []
    seen_task_ids: set[str] = set()
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"Task at index {index} must be an object.")
        task = _parse_task(record, index)
        if task.task_id in seen_task_ids:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        seen_task_ids.add(task.task_id)
        tasks.append(task)
    return tasks


def _parse_task(record: dict[str, Any], index: int) -> EvalTask:
    task_id = _required(record, "task_id", f"index {index}")
    prompt = _required(record, "prompt", task_id)
    requires_tools = _required(record, "requires_tools", task_id)
    expected_raw = _required(record, "expected", task_id)

    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Task at index {index} has invalid task_id.")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Task {task_id} has invalid prompt.")
    if not isinstance(requires_tools, bool):
        raise ValueError(f"Task {task_id} has invalid requires_tools.")
    if not isinstance(expected_raw, dict):
        raise ValueError(f"Task {task_id} has invalid expected.")

    max_tool_calls = record.get("max_tool_calls")
    if max_tool_calls is not None and (not isinstance(max_tool_calls, int) or max_tool_calls < 0):
        raise ValueError(f"Task {task_id} has invalid max_tool_calls.")

    tags = record.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"Task {task_id} has invalid tags.")

    return EvalTask(
        task_id=task_id,
        prompt=prompt,
        requires_tools=requires_tools,
        expected=_parse_expected(expected_raw, task_id),
        max_tool_calls=max_tool_calls,
        tags=tags,
    )


def _parse_expected(raw: dict[str, Any], task_id: str) -> TaskExpectations:
    tool_calls_raw = raw.get("tool_calls", [])
    answer_contains = raw.get("answer_contains", [])

    if not isinstance(tool_calls_raw, list):
        raise ValueError(f"Task {task_id} has invalid expected.tool_calls.")
    if not isinstance(answer_contains, list) or not all(isinstance(value, str) for value in answer_contains):
        raise ValueError(f"Task {task_id} has invalid expected.answer_contains.")

    tool_calls: list[ExpectedToolCall] = []
    for index, call in enumerate(tool_calls_raw):
        if not isinstance(call, dict):
            raise ValueError(f"Task {task_id} expected.tool_calls[{index}] must be an object.")
        tool_name = call.get("tool_name")
        arguments = call.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"Task {task_id} expected.tool_calls[{index}] has invalid tool_name.")
        if not isinstance(arguments, dict):
            raise ValueError(f"Task {task_id} expected.tool_calls[{index}] has invalid arguments.")
        tool_calls.append(ExpectedToolCall(tool_name=tool_name, arguments=arguments))

    return TaskExpectations(tool_calls=tool_calls, answer_contains=answer_contains)


def _required(record: dict[str, Any], field: str, task_label: str) -> Any:
    if field not in record:
        raise ValueError(f"Task {task_label} is missing required field: {field}")
    return record[field]
