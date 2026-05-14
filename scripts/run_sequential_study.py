#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.bootstrap import RuntimeBootstrap
from tool_trace_rag.config import AGENT_MAX_TOOL_CALLS, CUSTOMER_SUPPORT_DATA_PATH, EVAL_TASKS_PATH
from tool_trace_rag.eval.dataset import load_eval_tasks
from tool_trace_rag.experiments.sequential import SequentialStudyConfig, SequentialStudyRunner
from tool_trace_rag.memory.online import OnlineMemoryConfig, OnlineMemoryRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sequential online-memory study and write runtime artifacts under runs/.")
    parser.add_argument("--dataset", default=EVAL_TASKS_PATH, help="Path to evaluation task dataset.")
    parser.add_argument("--data", default=CUSTOMER_SUPPORT_DATA_PATH, help="Path to mock customer support data.")
    parser.add_argument("--output-dir", required=True, help="Runtime output directory, usually runs/sequential/<sequence_id>.")
    parser.add_argument("--sequence-id", default=None, help="Sequence id. Defaults to output directory name.")
    parser.add_argument("--ordering", choices=["original", "seeded_shuffle"], default="original", help="Task ordering mode.")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes through the dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for seeded_shuffle ordering.")
    parser.add_argument("--max-tool-calls", type=int, default=AGENT_MAX_TOOL_CALLS, help="Maximum tool calls per task.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    online_runner = OnlineMemoryRunner(bootstrap=RuntimeBootstrap())
    config = SequentialStudyConfig(
        sequence_id=args.sequence_id or output.name,
        output_dir=str(output),
        ordering=args.ordering,
        passes=args.passes,
        seed=args.seed,
        initial_corpus_size=0,
    )

    class _Wrapper:
        def __init__(self, runner: OnlineMemoryRunner, data: str, max_tool_calls: int) -> None:
            self.runner = runner
            self.data = data
            self.max_tool_calls = max_tool_calls

        def run(self, task: str, task_id: str | None = None):
            del task_id
            return self.runner.run(
                task,
                data_path=self.data,
                max_tool_calls=self.max_tool_calls,
                config=OnlineMemoryConfig(enabled=True, use_memory=True),
            )

    wrapped = _Wrapper(online_runner, args.data, args.max_tool_calls)
    result = SequentialStudyRunner(online_runner=wrapped).run(load_eval_tasks(args.dataset), config)
    print(f"Sequential study: {result['sequence_id']}")
    print(f"Total steps: {result['total_steps']}")
    print(f"Final corpus size: {result['final_corpus_size']}")
    print(f"Artifacts written: {result['output_dir']}")


if __name__ == "__main__":
    main()
