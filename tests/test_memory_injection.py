from tool_trace_rag.agent import ToolCallingAgent
from tool_trace_rag.memory.injection import MemoryPromptContext
from tool_trace_rag.providers.base import AssistantMessage
from tool_trace_rag.tools.registry import ToolRegistry


class CaptureProvider:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self):
        self.messages = None

    def complete(self, messages, tools, tool_choice):
        self.messages = messages
        return AssistantMessage(content="Done", tool_calls=[])


def test_agent_injects_memory_section_when_context_present():
    provider = CaptureProvider()
    context = MemoryPromptContext(prompt_section="Relevant past tool-use examples:\nExample 1", metadata={"enabled": True})
    agent = ToolCallingAgent(provider=provider, tools=ToolRegistry(), memory_context=context)

    trace = agent.run("Current task")

    assert "Relevant past tool-use examples" in provider.messages[0]["content"]
    assert trace.retrieval == {"enabled": True}


def test_agent_omits_memory_section_when_context_absent():
    provider = CaptureProvider()
    agent = ToolCallingAgent(provider=provider, tools=ToolRegistry())

    trace = agent.run("Current task")

    assert "Relevant past tool-use examples" not in provider.messages[0]["content"]
    assert trace.retrieval is None
