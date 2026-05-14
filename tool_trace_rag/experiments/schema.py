from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tool_trace_rag.eval.schema import EvalMetrics


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    dataset_path: str
    data_path: str
    output_dir: str
    max_tool_calls: int
    memory_trace_dir: str
    memory_vector_dir: str
    collection_name: str
    top_k: int
    memory_filter: str
    memory_strict: bool


@dataclass(frozen=True, slots=True)
class ArmSummary:
    name: str
    metrics: EvalMetrics
    trace_dir: str


@dataclass(frozen=True, slots=True)
class PairedTaskResult:
    task_id: str
    prompt: str
    requires_tools: bool
    baseline_passed: bool
    retrieval_passed: bool
    baseline_reasons: list[str]
    retrieval_reasons: list[str]
    baseline_tool_calls: int
    retrieval_tool_calls: int
    tool_call_delta: int
    baseline_final_answer: str
    retrieval_final_answer: str
    retrieval_metadata: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ExperimentComparison:
    success_rate_delta: float
    avg_tool_calls_delta: float
    over_tooling_rate_delta: float
    under_tooling_rate_delta: float
    retrieval_wins: int
    baseline_wins: int
    both_passed: int
    both_failed: int
    mean_tool_call_delta: float


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config: ExperimentConfig
    baseline: ArmSummary
    retrieval: ArmSummary
    comparison: ExperimentComparison
    paired_results: list[PairedTaskResult]


def to_jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
