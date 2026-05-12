from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.providers.base import AssistantMessage, ToolCall
from tool_trace_rag.providers.fake import FakeProvider
from tool_trace_rag.tools.registry import ToolDefinition, ToolRegistry


def build_test_registry():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="find_customer",
            description="Find a customer.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            function=lambda query: {"status": "found", "customer_id": "cust_001", "query": query},
        )
    )
    registry.register(
        ToolDefinition(
            name="get_customer_orders",
            description="Get orders.",
            parameters={"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
            function=lambda customer_id: {"status": "found", "orders": [{"order_id": "ord_1001"}]},
        )
    )
    return registry


def test_agent_supports_zero_tool_run():
    provider = FakeProvider([AssistantMessage(content="Return window is 30 days.", tool_calls=[])])
    agent = ToolCallingAgent(provider=provider, tools=build_test_registry())

    trace = agent.run("What is the return window?")

    assert trace.final_answer == "Return window is 30 days."
    assert trace.tool_calls == []
    assert trace.error is None
    assert trace.provider == "fake"
    assert trace.model == "fake-model"


def test_agent_executes_single_tool_then_final_answer():
    provider = FakeProvider(
        [
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_1", name="find_customer", arguments={"query": "Maya Chen"})]),
            AssistantMessage(content="Maya was found.", tool_calls=[]),
        ]
    )
    agent = ToolCallingAgent(provider=provider, tools=build_test_registry())

    trace = agent.run("Find Maya Chen")

    assert trace.final_answer == "Maya was found."
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "find_customer"
    assert trace.tool_calls[0].arguments == {"query": "Maya Chen"}
    assert trace.tool_calls[0].result["customer_id"] == "cust_001"


def test_agent_executes_multiple_tool_calls_in_order():
    provider = FakeProvider(
        [
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_1", name="find_customer", arguments={"query": "Maya Chen"})]),
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_2", name="get_customer_orders", arguments={"customer_id": "cust_001"})]),
            AssistantMessage(content="Maya has order ord_1001.", tool_calls=[]),
        ]
    )
    agent = ToolCallingAgent(provider=provider, tools=build_test_registry())

    trace = agent.run("Find Maya's orders")

    assert [call.tool_name for call in trace.tool_calls] == ["find_customer", "get_customer_orders"]
    assert trace.final_answer == "Maya has order ord_1001."


def test_agent_records_unknown_tool_as_tool_error():
    provider = FakeProvider(
        [
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_1", name="missing", arguments={})]),
            AssistantMessage(content="I could not use that tool.", tool_calls=[]),
        ]
    )
    agent = ToolCallingAgent(provider=provider, tools=build_test_registry())

    trace = agent.run("Use missing tool")

    assert trace.tool_calls[0].error == "UNKNOWN_TOOL"
    assert trace.tool_calls[0].result["status"] == "tool_error"


def test_agent_stops_at_max_tool_calls():
    provider = FakeProvider(
        [
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_1", name="find_customer", arguments={"query": "Maya Chen"})]),
            AssistantMessage(content=None, tool_calls=[ToolCall(id="call_2", name="find_customer", arguments={"query": "Maya Chen"})]),
        ]
    )
    agent = ToolCallingAgent(provider=provider, tools=build_test_registry(), max_tool_calls=1)

    trace = agent.run("Loop")

    assert trace.final_answer == ""
    assert trace.error == "max_tool_calls_exceeded"
    assert len(trace.tool_calls) == 1
