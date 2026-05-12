#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.cli import format_trace_summary
from tool_trace_rag.providers.openai_compatible import OpenAICompatibleProvider
from tool_trace_rag.tools.customer_support import build_customer_support_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one tool-calling agent task.")
    parser.add_argument("task", help="Task for the agent to perform.")
    parser.add_argument("--data", default="data/mock_customer_support.json", help="Path to mock customer support data.")
    parser.add_argument("--max-tool-calls", type=int, default=8, help="Maximum tool calls before aborting.")
    args = parser.parse_args()

    provider = OpenAICompatibleProvider.from_env()
    tools = build_customer_support_registry(args.data)
    agent = ToolCallingAgent(provider=provider, tools=tools, max_tool_calls=args.max_tool_calls)
    trace = agent.run(args.task)
    print(format_trace_summary(trace))


if __name__ == "__main__":
    main()
