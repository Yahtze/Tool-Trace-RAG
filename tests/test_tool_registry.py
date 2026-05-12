from tool_trace_rag.tools.registry import ToolDefinition, ToolRegistry


def test_registry_exposes_openai_tool_schemas():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo a value.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            function=lambda value: {"status": "ok", "value": value},
        )
    )

    assert registry.tool_schemas() == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def test_registry_executes_tool_by_name():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo a value.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda value: {"status": "ok", "value": value},
        )
    )

    result = registry.execute("echo", {"value": "hello"})

    assert result == {"status": "ok", "value": "hello"}


def test_registry_returns_structured_error_for_unknown_tool():
    registry = ToolRegistry()

    result = registry.execute("missing_tool", {})

    assert result == {
        "status": "tool_error",
        "error_code": "UNKNOWN_TOOL",
        "message": "Tool 'missing_tool' is not registered.",
    }


def test_registry_catches_tool_exceptions():
    def broken_tool():
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="broken",
            description="Breaks.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=broken_tool,
        )
    )

    result = registry.execute("broken", {})

    assert result == {
        "status": "tool_error",
        "error_code": "EXCEPTION",
        "message": "boom",
    }
