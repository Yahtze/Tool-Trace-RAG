import json

import pytest

from tool_trace_rag.eval.dataset import load_eval_tasks


def test_load_eval_tasks_parses_optional_defaults(tmp_path):
    dataset_path = tmp_path / "tasks.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "m2_test_001",
                    "prompt": "What is the return window?",
                    "requires_tools": False,
                    "expected": {"answer_contains": ["30 days"]},
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_eval_tasks(dataset_path)

    assert len(tasks) == 1
    assert tasks[0].task_id == "m2_test_001"
    assert tasks[0].prompt == "What is the return window?"
    assert tasks[0].requires_tools is False
    assert tasks[0].expected.tool_calls == []
    assert tasks[0].expected.answer_contains == ["30 days"]
    assert tasks[0].max_tool_calls is None
    assert tasks[0].tags == []


def test_load_eval_tasks_rejects_duplicate_task_ids(tmp_path):
    dataset_path = tmp_path / "tasks.json"
    record = {
        "task_id": "m2_dup",
        "prompt": "Find Maya Chen.",
        "requires_tools": True,
        "expected": {"tool_calls": [{"tool_name": "find_customer", "arguments": {"query": "Maya Chen"}}]},
    }
    dataset_path.write_text(json.dumps([record, record]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate task_id: m2_dup"):
        load_eval_tasks(dataset_path)


def test_load_eval_tasks_rejects_missing_required_fields(tmp_path):
    dataset_path = tmp_path / "tasks.json"
    dataset_path.write_text(json.dumps([{"task_id": "m2_bad"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="Task m2_bad is missing required field: prompt"):
        load_eval_tasks(dataset_path)


def test_milestone_02_dataset_meets_coverage_requirements():
    tasks = load_eval_tasks("data/eval_tasks_milestone_02.json")

    assert len(tasks) >= 30
    tool_required = [task for task in tasks if task.requires_tools]
    no_tool = [task for task in tasks if not task.requires_tools]
    assert len(tool_required) / len(tasks) >= 0.60
    assert len(tool_required) / len(tasks) <= 0.70
    assert len(no_tool) / len(tasks) >= 0.30
    assert len(no_tool) / len(tasks) <= 0.40

    tags = {tag for task in tasks for tag in task.tags}
    assert "refund" in tags
    assert "lookup" in tags
    assert "no_tool" in tags
    assert "edge_case" in tags
    assert "missing_customer" in tags
    assert "missing_order" in tags
    assert "ambiguous" in tags
