from tool_trace_rag.providers.openai_compatible import OpenAICompatibleProvider


def test_provider_normalizes_chat_completion_response_with_tool_call():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "find_customer",
                                "arguments": "{\"query\": \"Maya Chen\"}",
                            },
                        }
                    ],
                }
            }
        ]
    }

    message = OpenAICompatibleProvider._parse_response(payload)

    assert message.content is None
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].name == "find_customer"
    assert message.tool_calls[0].arguments == {"query": "Maya Chen"}


def test_provider_normalizes_final_answer_response():
    payload = {"choices": [{"message": {"content": "Final answer", "tool_calls": []}}]}

    message = OpenAICompatibleProvider._parse_response(payload)

    assert message.content == "Final answer"
    assert message.tool_calls == []


def test_provider_rejects_invalid_tool_arguments_json():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "find_customer", "arguments": "not-json"},
                        }
                    ],
                }
            }
        ]
    }

    try:
        OpenAICompatibleProvider._parse_response(payload)
    except ValueError as exc:
        assert str(exc) == "Tool call 'call_1' arguments are not valid JSON."
    else:
        raise AssertionError("Expected ValueError")
