from __future__ import annotations

import json
from pathlib import Path

from tool_trace_rag.eval.schema import EvalTask, TaskExpectations
from tool_trace_rag.experiments.sequential import SequentialStudyConfig, SequentialStudyRunner, ordered_tasks
from tool_trace_rag.memory.online import OnlineMemoryRunResult
from tool_trace_rag.traces.schema import AgentRunTrace


def sample_task(task_id: str, prompt: str) -> EvalTask:
    return EvalTask(
        task_id=task_id,
        prompt=prompt,
        requires_tools=False,
        expected=TaskExpectations(tool_calls=[], answer_contains=["ok"]),
        max_tool_calls=None,
        tags=["unit"],
    )


def test_ordered_tasks_supports_original_repeated_and_seeded_shuffle():
    tasks = [sample_task("a", "A"), sample_task("b", "B"), sample_task("c", "C")]

    original = ordered_tasks(tasks, ordering="original", passes=2, seed=7)
    shuffled_1 = ordered_tasks(tasks, ordering="seeded_shuffle", passes=1, seed=7)
    shuffled_2 = ordered_tasks(tasks, ordering="seeded_shuffle", passes=1, seed=7)

    assert [(item.pass_index, item.task.task_id) for item in original] == [(1, "a"), (1, "b"), (1, "c"), (2, "a"), (2, "b"), (2, "c")]
    assert [item.task.task_id for item in shuffled_1] == [item.task.task_id for item in shuffled_2]
    assert sorted(item.task.task_id for item in shuffled_1) == ["a", "b", "c"]


class FakeOnlineRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, task: str, task_id: str | None = None):
        self.calls.append(task_id or task)
        trace = AgentRunTrace(
            task_id=task_id,
            task=task,
            messages=[],
            tool_calls=[],
            final_answer="ok",
            success=True,
            provider="fake",
            model="fake",
            retrieval={"injected_count": max(0, len(self.calls) - 1)},
        )
        return OnlineMemoryRunResult(
            trace=trace,
            lifecycle={
                "retrieved_count": max(0, len(self.calls) - 1),
                "persisted_trace_path": f"trace-{len(self.calls)}.json",
                "upserted": True,
            },
        )


def test_sequential_study_runner_writes_runtime_artifacts(tmp_path: Path):
    tasks = [sample_task("a", "A"), sample_task("b", "B")]
    config = SequentialStudyConfig(
        sequence_id="seq-unit",
        output_dir=str(tmp_path / "runs" / "sequential" / "seq-unit"),
        ordering="original",
        passes=1,
        seed=123,
        initial_corpus_size=0,
    )

    result = SequentialStudyRunner(online_runner=FakeOnlineRunner()).run(tasks, config)

    output = Path(config.output_dir)
    assert result["sequence_id"] == "seq-unit"
    assert result["total_steps"] == 2
    assert (output / "sequence_config.json").exists()
    assert (output / "steps.jsonl").exists()
    lines = output.joinpath("steps.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["corpus_size_before"] == 0
    assert first["corpus_size_after"] == 1
    assert second["retrieved_memory_count"] == 1
