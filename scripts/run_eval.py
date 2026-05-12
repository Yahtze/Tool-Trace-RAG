#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.config import AGENT_MAX_TOOL_CALLS, CUSTOMER_SUPPORT_DATA_PATH, EVAL_TASKS_PATH
from tool_trace_rag.eval.dataset import load_eval_tasks
from tool_trace_rag.eval.evaluator import evaluate_tasks, score_trace, summarize_scores
from tool_trace_rag.eval.formatting import format_eval_report
from tool_trace_rag.eval.schema import EvalReport, TaskScore
from tool_trace_rag.providers.openai_compatible import OpenAICompatibleProvider
from tool_trace_rag.tools.customer_support import build_customer_support_registry
from tool_trace_rag.traces.store import DEFAULT_TRACE_DIR, TraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the milestone 2 evaluation baseline.")
    parser.add_argument("--tasks", default=EVAL_TASKS_PATH, help="Path to evaluation task dataset.")
    parser.add_argument("--data", default=CUSTOMER_SUPPORT_DATA_PATH, help="Path to mock customer support data.")
    parser.add_argument("--max-tool-calls", type=int, default=AGENT_MAX_TOOL_CALLS, help="Default maximum tool calls before aborting.")
    parser.add_argument("--verbose", action="store_true", help="Print per-task progress and elapsed time.")
    parser.add_argument("--save-traces", action="store_true", help="Write one JSON trace per evaluated task.")
    parser.add_argument("--trace-dir", default=None, help="Directory for persisted traces. Implies --save-traces.")
    args = parser.parse_args()

    trace_store = TraceStore(args.trace_dir or DEFAULT_TRACE_DIR) if args.save_traces or args.trace_dir else None
    traces_written = 0

    tasks = load_eval_tasks(args.tasks)

    def agent_factory(max_tool_calls: int) -> ToolCallingAgent:
        provider = OpenAICompatibleProvider.from_env()
        tools = build_customer_support_registry(args.data)
        return ToolCallingAgent(provider=provider, tools=tools, max_tool_calls=max_tool_calls)

    if args.verbose:
        scores: list[TaskScore] = []
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            max_tool_calls = task.max_tool_calls if task.max_tool_calls is not None else args.max_tool_calls
            started = time.perf_counter()
            agent = agent_factory(max_tool_calls)
            trace = agent.run(task.prompt, task_id=task.task_id)
            if trace_store is not None:
                trace_store.write_trace(trace)
                traces_written += 1
            elapsed = time.perf_counter() - started
            scores.append(score_trace(task, trace))
            print(f"[{index}/{total}] {task.task_id} completed in {elapsed:.2f}s")
        report = EvalReport(scores=scores, metrics=summarize_scores(scores))
    else:
        report = evaluate_tasks(
            tasks,
            agent_factory=agent_factory,
            default_max_tool_calls=args.max_tool_calls,
            trace_store=trace_store,
        )
        traces_written = len(report.scores) if trace_store is not None else 0

    print(format_eval_report(report))
    if trace_store is not None:
        print(f"Traces written: {traces_written}")
        print(f"Trace directory: {trace_store.root}")


if __name__ == "__main__":
    main()
