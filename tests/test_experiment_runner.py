from __future__ import annotations

import json
from pathlib import Path

from tool_trace_rag.eval.schema import EvalMetrics, EvalTask, TaskExpectations
from tool_trace_rag.experiments.runner import ExperimentRunner, compare_arm_metrics
from tool_trace_rag.experiments.schema import ArmSummary, ExperimentConfig
from tool_trace_rag.traces.schema import AgentRunTrace
from tool_trace_rag.traces.store import TraceStore


def metrics(success_rate: float, avg_tool_calls: float, over: float = 0.0, under: float = 0.0) -> EvalMetrics:
    return EvalMetrics(
        total_tasks=2,
        passed_tasks=int(success_rate * 2),
        failed_tasks=2 - int(success_rate * 2),
        success_rate=success_rate,
        tool_call_rate=0.5,
        avg_tool_calls=avg_tool_calls,
        avg_tool_calls_when_tools_required=avg_tool_calls,
        avg_tool_calls_when_tools_not_required=0.0,
        over_tooling_rate=over,
        under_tooling_rate=under,
        max_tool_call_violations=0,
        duplicate_tool_call_rate=0.0,
        error_rate=0.0,
    )


def test_compare_arm_metrics_computes_required_deltas():
    baseline = ArmSummary(name="baseline", metrics=metrics(0.5, 2.0, over=0.25, under=0.5), trace_dir="out/baseline/traces")
    retrieval = ArmSummary(name="retrieval", metrics=metrics(1.0, 1.5, over=0.0, under=0.25), trace_dir="out/retrieval/traces")

    comparison = compare_arm_metrics(
        baseline=baseline,
        retrieval=retrieval,
        paired_outcomes=[
            {"baseline_passed": False, "retrieval_passed": True, "tool_call_delta": -1},
            {"baseline_passed": True, "retrieval_passed": True, "tool_call_delta": 0},
        ],
    )

    assert comparison.success_rate_delta == 0.5
    assert comparison.avg_tool_calls_delta == -0.5
    assert comparison.over_tooling_rate_delta == -0.25
    assert comparison.under_tooling_rate_delta == -0.25
    assert comparison.retrieval_wins == 1
    assert comparison.baseline_wins == 0
    assert comparison.both_passed == 1
    assert comparison.both_failed == 0
    assert comparison.mean_tool_call_delta == -0.5


def sample_task(task_id: str = "task-1") -> EvalTask:
    return EvalTask(
        task_id=task_id,
        prompt="Can Maya return her headphones?",
        requires_tools=False,
        expected=TaskExpectations(tool_calls=[], answer_contains=["yes"]),
        max_tool_calls=None,
        tags=["unit"],
    )


class FakeAgent:
    def __init__(self, trace: AgentRunTrace) -> None:
        self.trace = trace

    def run(self, task: str, task_id: str | None = None) -> AgentRunTrace:
        return self.trace


class FakeBootstrap:
    def __init__(self) -> None:
        self.memory_context_tasks: list[str] = []
        self.agent_memory_contexts: list[object | None] = []
        self.trace_dirs: list[str] = []

    def build_memory_context(self, **kwargs):
        self.memory_context_tasks.append(kwargs["task"])
        return type("MemoryContext", (), {"metadata": {"enabled": True, "injected_count": 1, "memories": [{"trace_id": "m1"}], "error": None}})()

    def build_agent(self, *, data_path: str, max_tool_calls: int, memory_context=None):
        self.agent_memory_contexts.append(memory_context)
        final = "yes with memory" if memory_context is not None else "yes baseline"
        return FakeAgent(
            AgentRunTrace(
                task_id="task-1",
                task="Can Maya return her headphones?",
                messages=[{"role": "user", "content": "Can Maya return her headphones?"}],
                tool_calls=[],
                final_answer=final,
                success=True,
                provider="fake",
                model="fake-model",
                retrieval=getattr(memory_context, "metadata", None),
            )
        )

    def build_trace_store(self, trace_dir):
        self.trace_dirs.append(str(trace_dir))
        return TraceStore(trace_dir)


def test_experiment_runner_executes_baseline_and_retrieval_with_isolation(tmp_path: Path):
    bootstrap = FakeBootstrap()
    config = ExperimentConfig(
        experiment_id="exp-unit",
        dataset_path="data/eval_tasks_milestone_02.json",
        data_path="data/mock_customer_support.json",
        output_dir=str(tmp_path / "exp-unit"),
        max_tool_calls=8,
        memory_trace_dir=str(tmp_path / "memory-traces"),
        memory_vector_dir=str(tmp_path / "memory-vectors"),
        collection_name="test_collection",
        top_k=3,
        memory_filter="successful_only",
        memory_strict=False,
    )

    result = ExperimentRunner(bootstrap=bootstrap).run(tasks=[sample_task()], config=config)

    assert bootstrap.agent_memory_contexts[0] is None
    assert bootstrap.agent_memory_contexts[1] is not None
    assert bootstrap.memory_context_tasks == ["Can Maya return her headphones?"]
    assert result.baseline.metrics.success_rate == 1.0
    assert result.retrieval.metrics.success_rate == 1.0
    assert result.paired_results[0].baseline_final_answer == "yes baseline"
    assert result.paired_results[0].retrieval_final_answer == "yes with memory"
    assert result.paired_results[0].retrieval_metadata == {"enabled": True, "injected_count": 1, "memories": [{"trace_id": "m1"}], "error": None}
    assert "baseline/traces" in result.baseline.trace_dir
    assert "retrieval/traces" in result.retrieval.trace_dir


def test_experiment_runner_writes_required_artifacts(tmp_path: Path):
    bootstrap = FakeBootstrap()
    output_dir = tmp_path / "exp-artifacts"
    config = ExperimentConfig(
        experiment_id="exp-artifacts",
        dataset_path="data/eval_tasks_milestone_02.json",
        data_path="data/mock_customer_support.json",
        output_dir=str(output_dir),
        max_tool_calls=8,
        memory_trace_dir=str(tmp_path / "memory-traces"),
        memory_vector_dir=str(tmp_path / "memory-vectors"),
        collection_name="tool_trace_memory",
        top_k=3,
        memory_filter="successful_only",
        memory_strict=False,
    )

    ExperimentRunner(bootstrap=bootstrap).run(tasks=[sample_task()], config=config)

    assert (output_dir / "experiment_config.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "paired_results.jsonl").exists()
    assert (output_dir / "baseline" / "summary.json").exists()
    assert (output_dir / "retrieval" / "summary.json").exists()

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["experiment_id"] == "exp-artifacts"
    assert summary["baseline"]["name"] == "baseline"
    assert summary["retrieval"]["name"] == "retrieval"
    assert "success_rate_delta" in summary["comparison"]

    config_payload = json.loads((output_dir / "experiment_config.json").read_text())
    assert config_payload["memory_trace_dir"] == str(tmp_path / "memory-traces")
    assert config_payload["memory_vector_dir"] == str(tmp_path / "memory-vectors")

    lines = (output_dir / "paired_results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task_id"] == "task-1"


def test_experiment_runner_does_not_build_vector_store_or_online_memory(tmp_path: Path):
    class GuardedBootstrap(FakeBootstrap):
        def build_vector_store(self, *args, **kwargs):
            raise AssertionError("experiment runner must not mutate or open vector store directly")

    bootstrap = GuardedBootstrap()
    config = ExperimentConfig(
        experiment_id="exp-no-mutation",
        dataset_path="data/eval_tasks_milestone_02.json",
        data_path="data/mock_customer_support.json",
        output_dir=str(tmp_path / "exp-no-mutation"),
        max_tool_calls=8,
        memory_trace_dir=str(tmp_path / "memory-traces"),
        memory_vector_dir=str(tmp_path / "memory-vectors"),
        collection_name="tool_trace_memory",
        top_k=3,
        memory_filter="successful_only",
        memory_strict=False,
    )

    result = ExperimentRunner(bootstrap=bootstrap).run(tasks=[sample_task()], config=config)

    assert result.config.memory_trace_dir == str(tmp_path / "memory-traces")
    assert result.config.memory_vector_dir == str(tmp_path / "memory-vectors")
    assert result.paired_results[0].retrieval_metadata is not None
