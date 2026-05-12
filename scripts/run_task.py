#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.config import AGENT_MAX_TOOL_CALLS, CUSTOMER_SUPPORT_DATA_PATH
from tool_trace_rag.cli import format_trace_summary
from tool_trace_rag.providers.openai_compatible import OpenAICompatibleProvider
from tool_trace_rag.tools.customer_support import build_customer_support_registry
from tool_trace_rag.traces.store import DEFAULT_TRACE_DIR, TraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one tool-calling agent task.")
    parser.add_argument("task", help="Task for the agent to perform.")
    parser.add_argument("--data", default=CUSTOMER_SUPPORT_DATA_PATH, help="Path to mock customer support data.")
    parser.add_argument("--max-tool-calls", type=int, default=AGENT_MAX_TOOL_CALLS, help="Maximum tool calls before aborting.")
    parser.add_argument("--save-trace", action="store_true", help="Write the completed run trace as JSON.")
    parser.add_argument("--trace-dir", default=None, help="Directory for persisted trace. Implies --save-trace.")
    args = parser.parse_args()

    provider = OpenAICompatibleProvider.from_env()
    tools = build_customer_support_registry(args.data)
    agent = ToolCallingAgent(provider=provider, tools=tools, max_tool_calls=args.max_tool_calls)
    trace = agent.run(args.task)
    print(format_trace_summary(trace))

    if args.save_trace or args.trace_dir:
        store = TraceStore(args.trace_dir or DEFAULT_TRACE_DIR)
        path = store.write_trace(trace)
        print(f"Trace written: {path}")


if __name__ == "__main__":
    main()
