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
    MEMORY_FILTER,
    MEMORY_STRICT,
    MEMORY_TOP_K,
    TRACE_DIR,
    USE_MEMORY,
    VECTOR_COLLECTION_NAME,
    VECTOR_DIR,
)
from tool_trace_rag.cli import format_trace_summary
from tool_trace_rag.traces.store import DEFAULT_TRACE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one tool-calling agent task.")
    parser.add_argument("task", help="Task for the agent to perform.")
    parser.add_argument("--data", default=CUSTOMER_SUPPORT_DATA_PATH, help="Path to mock customer support data.")
    parser.add_argument("--max-tool-calls", type=int, default=AGENT_MAX_TOOL_CALLS, help="Maximum tool calls before aborting.")
    parser.add_argument("--save-trace", action="store_true", help="Write the completed run trace as JSON.")
    parser.add_argument("--trace-dir", default=None, help="Directory for persisted trace. Implies --save-trace.")
    parser.add_argument("--use-memory", action="store_true", default=USE_MEMORY, help="Retrieve and inject prior trace memories.")
    parser.add_argument("--vector-dir", default=str(VECTOR_DIR), help="Directory for the local vector store.")
    parser.add_argument("--collection", default=VECTOR_COLLECTION_NAME, help="Vector collection name.")
    parser.add_argument("--top-k", type=int, default=MEMORY_TOP_K, help="Number of memory candidates to retrieve.")
    parser.add_argument("--memory-filter", default=MEMORY_FILTER, choices=["successful_only", "failed_only", "all"], help="Memory filtering policy.")
    parser.add_argument("--memory-strict", action="store_true", default=MEMORY_STRICT, help="Fail instead of falling back when retrieval fails.")
    args = parser.parse_args()

    bootstrap = RuntimeBootstrap()
    memory_context = None
    if args.use_memory:
        memory_context = bootstrap.build_memory_context(
            task=args.task,
            trace_dir=args.trace_dir or TRACE_DIR,
            vector_dir=args.vector_dir,
            collection_name=args.collection,
            top_k=args.top_k,
            memory_filter=args.memory_filter,
            strict=args.memory_strict,
        )
    agent = bootstrap.build_agent(data_path=args.data, max_tool_calls=args.max_tool_calls, memory_context=memory_context)
    trace = agent.run(args.task)
    print(format_trace_summary(trace))
    if trace.retrieval:
        print(f"Retrieved memories: {trace.retrieval['injected_count']}")
        for memory in trace.retrieval.get("memories", []):
            print(f"{memory['rank']}. trace_id={memory['trace_id']} task_id={memory['task_id']} score={memory['score']}")
        if trace.retrieval.get("error"):
            print(f"Memory retrieval error: {trace.retrieval['error']}")

    if args.save_trace or args.trace_dir:
        store = bootstrap.build_trace_store(args.trace_dir or DEFAULT_TRACE_DIR)
        path = store.write_trace(trace)
        print(f"Trace written: {path}")


if __name__ == "__main__":
    main()
