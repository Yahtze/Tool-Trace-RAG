#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.eval.dataset import load_eval_tasks
from tool_trace_rag.eval.evaluator import evaluate_tasks, score_trace, summarize_scores
from tool_trace_rag.eval.formatting import format_eval_report
from tool_trace_rag.eval.schema import EvalReport, TaskScore
from tool_trace_rag.providers.openai_compatible import OpenAICompatibleProvider
from tool_trace_rag.tools.customer_support import build_customer_support_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the milestone 2 evaluation baseline.")
    parser.add_argument("--tasks", default="data/eval_tasks_milestone_02.json", help="Path to evaluation task dataset.")
    parser.add_argument("--data", default="data/mock_customer_support.json", help="Path to mock customer support data.")
    parser.add_argument("--max-tool-calls", type=int, default=8, help="Default maximum tool calls before aborting.")
    parser.add_argument("--verbose", action="store_true", help="Print per-task progress and elapsed time.")
    args = parser.parse_args()

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
            elapsed = time.perf_counter() - started
            scores.append(score_trace(task, trace))
            print(f"[{index}/{total}] {task.task_id} completed in {elapsed:.2f}s")
        report = EvalReport(scores=scores, metrics=summarize_scores(scores))
    else:
        report = evaluate_tasks(tasks, agent_factory=agent_factory, default_max_tool_calls=args.max_tool_calls)

    print(format_eval_report(report))


if __name__ == "__main__":
    main()
