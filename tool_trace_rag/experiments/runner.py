from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tool_trace_rag.eval.evaluator import score_trace, summarize_scores
from tool_trace_rag.eval.schema import EvalTask, TaskScore
from tool_trace_rag.experiments.schema import (
    ArmSummary,
    ExperimentComparison,
    ExperimentConfig,
    ExperimentResult,
    PairedTaskResult,
)


class ExperimentRunner:
    def __init__(self, bootstrap: Any) -> None:
        self.bootstrap = bootstrap

    def run(self, tasks: list[EvalTask], config: ExperimentConfig) -> ExperimentResult:
        output_dir = Path(config.output_dir)
        baseline_trace_dir = output_dir / "baseline" / "traces"
        retrieval_trace_dir = output_dir / "retrieval" / "traces"
        baseline_scores: list[TaskScore] = []
        retrieval_scores: list[TaskScore] = []
        paired_results: list[PairedTaskResult] = []

        for task in tasks:
            max_tool_calls = task.max_tool_calls if task.max_tool_calls is not None else config.max_tool_calls

            baseline_agent = self.bootstrap.build_agent(data_path=config.data_path, max_tool_calls=max_tool_calls, memory_context=None)
            baseline_trace = baseline_agent.run(task.prompt, task_id=task.task_id)
            self.bootstrap.build_trace_store(baseline_trace_dir).write_trace(baseline_trace)
            baseline_score = score_trace(task, baseline_trace)
            baseline_scores.append(baseline_score)

            memory_context = self.bootstrap.build_memory_context(
                task=task.prompt,
                trace_dir=config.memory_trace_dir,
                vector_dir=config.memory_vector_dir,
                collection_name=config.collection_name,
                top_k=config.top_k,
                memory_filter=config.memory_filter,
                strict=config.memory_strict,
            )
            retrieval_agent = self.bootstrap.build_agent(data_path=config.data_path, max_tool_calls=max_tool_calls, memory_context=memory_context)
            retrieval_trace = retrieval_agent.run(task.prompt, task_id=task.task_id)
            self.bootstrap.build_trace_store(retrieval_trace_dir).write_trace(retrieval_trace)
            retrieval_score = score_trace(task, retrieval_trace)
            retrieval_scores.append(retrieval_score)

            paired_results.append(_paired_result(task, baseline_score, retrieval_score))

        baseline_summary = ArmSummary(name="baseline", metrics=summarize_scores(baseline_scores), trace_dir=str(baseline_trace_dir))
        retrieval_summary = ArmSummary(name="retrieval", metrics=summarize_scores(retrieval_scores), trace_dir=str(retrieval_trace_dir))
        comparison = compare_arm_metrics(
            baseline=baseline_summary,
            retrieval=retrieval_summary,
            paired_outcomes=[asdict(item) for item in paired_results],
        )
        result = ExperimentResult(
            config=config,
            baseline=baseline_summary,
            retrieval=retrieval_summary,
            comparison=comparison,
            paired_results=paired_results,
        )
        write_experiment_artifacts(result)
        return result


def _paired_result(task: EvalTask, baseline: TaskScore, retrieval: TaskScore) -> PairedTaskResult:
    baseline_calls = len(baseline.trace.tool_calls)
    retrieval_calls = len(retrieval.trace.tool_calls)
    return PairedTaskResult(
        task_id=task.task_id,
        prompt=task.prompt,
        requires_tools=task.requires_tools,
        baseline_passed=baseline.passed,
        retrieval_passed=retrieval.passed,
        baseline_reasons=baseline.reasons,
        retrieval_reasons=retrieval.reasons,
        baseline_tool_calls=baseline_calls,
        retrieval_tool_calls=retrieval_calls,
        tool_call_delta=retrieval_calls - baseline_calls,
        baseline_final_answer=baseline.trace.final_answer[:500],
        retrieval_final_answer=retrieval.trace.final_answer[:500],
        retrieval_metadata=retrieval.trace.retrieval,
    )


def compare_arm_metrics(
    baseline: ArmSummary,
    retrieval: ArmSummary,
    paired_outcomes: list[dict[str, object]],
) -> ExperimentComparison:
    total = len(paired_outcomes)
    tool_call_delta_sum = sum(int(item["tool_call_delta"]) for item in paired_outcomes)
    return ExperimentComparison(
        success_rate_delta=round(retrieval.metrics.success_rate - baseline.metrics.success_rate, 4),
        avg_tool_calls_delta=round(retrieval.metrics.avg_tool_calls - baseline.metrics.avg_tool_calls, 4),
        over_tooling_rate_delta=round(retrieval.metrics.over_tooling_rate - baseline.metrics.over_tooling_rate, 4),
        under_tooling_rate_delta=round(retrieval.metrics.under_tooling_rate - baseline.metrics.under_tooling_rate, 4),
        retrieval_wins=sum(1 for item in paired_outcomes if item["retrieval_passed"] and not item["baseline_passed"]),
        baseline_wins=sum(1 for item in paired_outcomes if item["baseline_passed"] and not item["retrieval_passed"]),
        both_passed=sum(1 for item in paired_outcomes if item["baseline_passed"] and item["retrieval_passed"]),
        both_failed=sum(1 for item in paired_outcomes if not item["baseline_passed"] and not item["retrieval_passed"]),
        mean_tool_call_delta=round(tool_call_delta_sum / total, 4) if total else 0.0,
    )


def write_experiment_artifacts(result: ExperimentResult) -> None:
    output_dir = Path(result.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval").mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "experiment_config.json", asdict(result.config))
    _write_json(
        output_dir / "summary.json",
        {
            "experiment_id": result.config.experiment_id,
            "baseline": asdict(result.baseline),
            "retrieval": asdict(result.retrieval),
            "comparison": asdict(result.comparison),
        },
    )
    _write_json(output_dir / "baseline" / "summary.json", asdict(result.baseline))
    _write_json(output_dir / "retrieval" / "summary.json", asdict(result.retrieval))
    with (output_dir / "paired_results.jsonl").open("w", encoding="utf-8") as handle:
        for item in result.paired_results:
            handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
