from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tool_trace_rag.eval.evaluator import score_trace
from tool_trace_rag.eval.schema import EvalTask


@dataclass(frozen=True, slots=True)
class SequentialStudyConfig:
    sequence_id: str
    output_dir: str
    ordering: str = "original"
    passes: int = 1
    seed: int = 0
    initial_corpus_size: int = 0


@dataclass(frozen=True, slots=True)
class OrderedTask:
    step: int
    pass_index: int
    task: EvalTask


def ordered_tasks(tasks: list[EvalTask], ordering: str, passes: int, seed: int) -> list[OrderedTask]:
    if passes < 1:
        raise ValueError("passes must be >= 1")
    ordered: list[OrderedTask] = []
    step = 1
    for pass_index in range(1, passes + 1):
        current = list(tasks)
        if ordering == "seeded_shuffle":
            rng = random.Random(seed + pass_index - 1)
            rng.shuffle(current)
        elif ordering != "original":
            raise ValueError(f"unsupported ordering: {ordering}")
        for task in current:
            ordered.append(OrderedTask(step=step, pass_index=pass_index, task=task))
            step += 1
    return ordered


class SequentialStudyRunner:
    def __init__(self, online_runner: Any) -> None:
        self.online_runner = online_runner

    def run(self, tasks: list[EvalTask], config: SequentialStudyConfig) -> dict[str, Any]:
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        corpus_size = config.initial_corpus_size
        rows: list[dict[str, Any]] = []
        for item in ordered_tasks(tasks, config.ordering, config.passes, config.seed):
            before = corpus_size
            result = self.online_runner.run(item.task.prompt, task_id=item.task.task_id)
            trace = result.trace
            lifecycle = dict(result.lifecycle)
            score = score_trace(item.task, trace)
            upsert_data = lifecycle.get("upsert", {}) if isinstance(lifecycle.get("upsert"), dict) else {}
            upserted = bool(upsert_data.get("upserted", lifecycle.get("upserted", False)))
            if upserted:
                corpus_size += 1
            retrieved_count = _retrieved_count(trace.retrieval, lifecycle)
            rows.append(
                {
                    "step": item.step,
                    "pass_index": item.pass_index,
                    "task_id": item.task.task_id,
                    "prompt": item.task.prompt,
                    "passed": score.passed,
                    "reasons": score.reasons,
                    "tool_calls": len(trace.tool_calls),
                    "over_tooling": any("expected zero tool calls" in reason for reason in score.reasons),
                    "under_tooling": any("missing expected tool call" in reason for reason in score.reasons),
                    "corpus_size_before": before,
                    "corpus_size_after": corpus_size,
                    "retrieved_memory_count": retrieved_count,
                    "retrieval_metadata": trace.retrieval,
                    "persisted_trace_path": _persisted_path(lifecycle),
                    "upserted": upserted,
                }
            )
        summary = {
            "sequence_id": config.sequence_id,
            "total_steps": len(rows),
            "output_dir": str(output),
            "final_corpus_size": corpus_size,
        }
        _write_json(output / "sequence_config.json", asdict(config))
        _write_json(output / "summary.json", summary)
        _write_jsonl(output / "steps.jsonl", rows)
        return summary


def _persisted_path(lifecycle: dict[str, Any]) -> str | None:
    if "persisted_trace_path" in lifecycle:
        return lifecycle.get("persisted_trace_path")
    persistence = lifecycle.get("persistence", {}) if isinstance(lifecycle.get("persistence"), dict) else {}
    value = persistence.get("trace_path")
    return str(value) if value else None


def _retrieved_count(retrieval: Any, lifecycle: dict[str, Any]) -> int:
    if "retrieved_count" in lifecycle:
        return int(lifecycle.get("retrieved_count") or 0)
    if isinstance(lifecycle.get("retrieval"), dict) and "retrieved_count" in lifecycle["retrieval"]:
        return int(lifecycle["retrieval"].get("retrieved_count") or 0)
    if isinstance(retrieval, dict):
        if "injected_count" in retrieval:
            return int(retrieval.get("injected_count") or 0)
        memories = retrieval.get("memories")
        if isinstance(memories, list):
            return len(memories)
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
