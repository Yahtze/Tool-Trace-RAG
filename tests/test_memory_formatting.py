from tool_trace_rag.memory.formatting import MemoryExample, format_memory_prompt_section, format_memory_snippet
from tool_trace_rag.traces.schema import AgentRunTrace, ToolCallTrace


def make_trace(long: bool = False) -> AgentRunTrace:
    text = "x" * 100 if long else "Maya can return delivered headphones."
    return AgentRunTrace(
        trace_id="trace-old",
        created_at="2026-05-12T00:00:00+00:00",
        task_id="old-task",
        task="Maya asks about returning headphones.",
        messages=[{"role": "user", "content": "hidden transcript should not be copied"}],
        tool_calls=[ToolCallTrace("call_1", "find_customer", {"query": "Maya Chen"}, {"status": "found"}, None, 1.0)],
        final_answer=text,
        success=True,
        provider="fake",
        model="fake-model",
    )


def test_format_memory_snippet_is_deterministic_and_observable_only():
    example = MemoryExample(rank=1, score=0.25, trace=make_trace(), source_path="old.json")

    snippet = format_memory_snippet(example, max_chars=1000)

    assert snippet == """Example 1:
Trace ID: trace-old
Task ID: old-task
Task: Maya asks about returning headphones.
Outcome: success
Tool path:
1. find_customer {\"query\": \"Maya Chen\"} -> {\"status\": \"found\"}
Final answer: Maya can return delivered headphones."""
    assert "hidden transcript" not in snippet


def test_format_memory_snippet_truncates_predictably():
    snippet = format_memory_snippet(MemoryExample(1, 0.0, make_trace(long=True), "old.json"), max_chars=160)

    assert len(snippet) == 160
    assert snippet.endswith("…")


def test_format_memory_prompt_section_wraps_examples_with_guidance():
    section = format_memory_prompt_section([MemoryExample(1, 0.25, make_trace(), "old.json")], max_chars_per_snippet=1000)

    assert section.startswith("Relevant past tool-use examples:\n\nExample 1:")
    assert "Use these examples only as guidance." in section
