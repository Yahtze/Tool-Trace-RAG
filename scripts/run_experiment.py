#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.bootstrap import RuntimeBootstrap
from tool_trace_rag.config import (
    AGENT_MAX_TOOL_CALLS,
    CUSTOMER_SUPPORT_DATA_PATH,
    EVAL_TASKS_PATH,
    MEMORY_FILTER,
    MEMORY_STRICT,
    MEMORY_TOP_K,
    TRACE_DIR,
    VECTOR_COLLECTION_NAME,
    VECTOR_DIR,
)
from tool_trace_rag.eval.dataset import load_eval_tasks
from tool_trace_rag.experiments.runner import ExperimentRunner
from tool_trace_rag.experiments.schema import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a controlled baseline-vs-retrieval memory experiment.")
    parser.add_argument("--dataset", default=EVAL_TASKS_PATH, help="Path to evaluation task dataset.")
    parser.add_argument("--data", default=CUSTOMER_SUPPORT_DATA_PATH, help="Path to mock customer support data.")
    parser.add_argument("--output-dir", required=True, help="Directory where experiment artifacts will be written.")
    parser.add_argument("--experiment-id", default=None, help="Stable experiment id. Defaults to output directory name.")
    parser.add_argument("--max-tool-calls", type=int, default=AGENT_MAX_TOOL_CALLS, help="Default maximum tool calls per task.")
    parser.add_argument("--memory-trace-dir", default=str(TRACE_DIR), help="Read-only trace corpus used by retrieval arm.")
    parser.add_argument("--memory-vector-dir", default=str(VECTOR_DIR), help="Read-only vector corpus used by retrieval arm.")
    parser.add_argument("--collection", default=VECTOR_COLLECTION_NAME, help="Vector collection name.")
    parser.add_argument("--top-k", type=int, default=MEMORY_TOP_K, help="Number of memory candidates to retrieve.")
    parser.add_argument("--memory-filter", default=MEMORY_FILTER, choices=["successful_only", "failed_only", "all"], help="Memory filtering policy.")
    parser.add_argument("--memory-strict", action="store_true", default=MEMORY_STRICT, help="Fail instead of falling back when retrieval fails.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    config = ExperimentConfig(
        experiment_id=args.experiment_id or output_dir.name,
        dataset_path=args.dataset,
        data_path=args.data,
        output_dir=str(output_dir),
        max_tool_calls=args.max_tool_calls,
        memory_trace_dir=args.memory_trace_dir,
        memory_vector_dir=args.memory_vector_dir,
        collection_name=args.collection,
        top_k=args.top_k,
        memory_filter=args.memory_filter,
        memory_strict=args.memory_strict,
    )
    tasks = load_eval_tasks(args.dataset)
    result = ExperimentRunner(bootstrap=RuntimeBootstrap()).run(tasks=tasks, config=config)

    print(f"Experiment: {result.config.experiment_id}")
    print(f"Dataset tasks: {result.baseline.metrics.total_tasks}")
    print("Arms: baseline, retrieval")
    print()
    print(f"Baseline success rate: {result.baseline.metrics.success_rate}")
    print(f"Retrieval success rate: {result.retrieval.metrics.success_rate}")
    print(f"Success rate delta: {result.comparison.success_rate_delta}")
    print(f"Average tool calls delta: {result.comparison.avg_tool_calls_delta}")
    print(f"Retrieval wins: {result.comparison.retrieval_wins}")
    print(f"Baseline wins: {result.comparison.baseline_wins}")
    print()
    print(f"Artifacts written: {result.config.output_dir}")


if __name__ == "__main__":
    main()
