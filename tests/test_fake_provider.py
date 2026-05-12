from tool_trace_rag.providers.base import AssistantMessage, ToolCall
from tool_trace_rag.providers.fake import FakeProvider


def test_fake_provider_returns_scripted_messages_in_order():
    provider = FakeProvider(
        responses=[
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_1", name="find_customer", arguments={"query": "Maya Chen"})]),
            AssistantMessage(content="Final answer", tool_calls=[]),
        ]
    )

    first = provider.complete(messages=[], tools=[])
    second = provider.complete(messages=[], tools=[])

    assert first.tool_calls[0].name == "find_customer"
    assert second.content == "Final answer"
    assert provider.call_count == 2


def test_fake_provider_raises_when_script_exhausted():
    provider = FakeProvider(responses=[])

    try:
        provider.complete(messages=[], tools=[])
    except RuntimeError as exc:
        assert str(exc) == "FakeProvider has no scripted responses left."
    else:
        raise AssertionError("Expected RuntimeError")
