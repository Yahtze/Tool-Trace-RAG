#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
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
    USE_MEMORY,
    VECTOR_COLLECTION_NAME,
    VECTOR_DIR,
)
from tool_trace_rag.eval.dataset import load_eval_tasks
from tool_trace_rag.eval.evaluator import evaluate_tasks, score_trace, summarize_scores
from tool_trace_rag.eval.formatting import format_eval_report
from tool_trace_rag.eval.schema import EvalReport, TaskScore
from tool_trace_rag.traces.store import DEFAULT_TRACE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the milestone 2 evaluation baseline.")
    parser.add_argument("--tasks", default=EVAL_TASKS_PATH, help="Path to evaluation task dataset.")
    parser.add_argument("--data", default=CUSTOMER_SUPPORT_DATA_PATH, help="Path to mock customer support data.")
    parser.add_argument("--max-tool-calls", type=int, default=AGENT_MAX_TOOL_CALLS, help="Default maximum tool calls before aborting.")
    parser.add_argument("--verbose", action="store_true", help="Print per-task progress and elapsed time.")
    parser.add_argument("--save-traces", action="store_true", help="Write one JSON trace per evaluated task.")
    parser.add_argument("--trace-dir", default=None, help="Directory for persisted traces. Implies --save-traces.")
    parser.add_argument("--use-memory", action="store_true", default=USE_MEMORY, help="Retrieve and inject prior trace memories.")
    parser.add_argument("--vector-dir", default=str(VECTOR_DIR), help="Directory for the local vector store.")
    parser.add_argument("--collection", default=VECTOR_COLLECTION_NAME, help="Vector collection name.")
    parser.add_argument("--top-k", type=int, default=MEMORY_TOP_K, help="Number of memory candidates to retrieve.")
    parser.add_argument("--memory-filter", default=MEMORY_FILTER, choices=["successful_only", "failed_only", "all"], help="Memory filtering policy.")
    parser.add_argument("--memory-strict", action="store_true", default=MEMORY_STRICT, help="Fail instead of falling back when retrieval fails.")
    args = parser.parse_args()

    bootstrap = RuntimeBootstrap()
    trace_store = bootstrap.build_trace_store(args.trace_dir or DEFAULT_TRACE_DIR) if args.save_traces or args.trace_dir else None
    traces_written = 0

    tasks = load_eval_tasks(args.tasks)
    base_factory = bootstrap.build_eval_agent_factory(data_path=args.data)

    def agent_factory(max_tool_calls: int, task_prompt: str | None = None):
        if not args.use_memory or task_prompt is None:
            return base_factory(max_tool_calls)
        memory_context = bootstrap.build_memory_context(
            task=task_prompt,
            trace_dir=args.trace_dir or TRACE_DIR,
            vector_dir=args.vector_dir,
            collection_name=args.collection,
            top_k=args.top_k,
            memory_filter=args.memory_filter,
            strict=args.memory_strict,
        )
        return bootstrap.build_agent(data_path=args.data, max_tool_calls=max_tool_calls, memory_context=memory_context)

    if args.verbose:
        scores: list[TaskScore] = []
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            max_tool_calls = task.max_tool_calls if task.max_tool_calls is not None else args.max_tool_calls
            started = time.perf_counter()
            agent = agent_factory(max_tool_calls, task.prompt)
            trace = agent.run(task.prompt, task_id=task.task_id)
            if trace_store is not None:
                trace_store.write_trace(trace)
                traces_written += 1
            elapsed = time.perf_counter() - started
            scores.append(score_trace(task, trace))
            print(f"[{index}/{total}] {task.task_id} completed in {elapsed:.2f}s")
        report = EvalReport(scores=scores, metrics=summarize_scores(scores))
    else:
        if args.use_memory:
            scores: list[TaskScore] = []
            for task in tasks:
                max_tool_calls = task.max_tool_calls if task.max_tool_calls is not None else args.max_tool_calls
                agent = agent_factory(max_tool_calls, task.prompt)
                trace = agent.run(task.prompt, task_id=task.task_id)
                if trace_store is not None:
                    trace_store.write_trace(trace)
                    traces_written += 1
                scores.append(score_trace(task, trace))
            report = EvalReport(scores=scores, metrics=summarize_scores(scores))
        else:
            report = evaluate_tasks(
                tasks,
                agent_factory=lambda max_tool_calls: agent_factory(max_tool_calls),
                default_max_tool_calls=args.max_tool_calls,
                trace_store=trace_store,
            )
            traces_written = len(report.scores) if trace_store is not None else 0

    print(format_eval_report(report))
    if args.use_memory:
        print("Memory enabled: true")
        print(f"Memory top_k: {args.top_k}")
    if trace_store is not None:
        print(f"Traces written: {traces_written}")
        print(f"Trace directory: {trace_store.root}")


if __name__ == "__main__":
    main()
