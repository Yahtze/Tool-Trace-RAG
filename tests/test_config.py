from __future__ import annotations

import importlib


def test_online_memory_config_defaults_false(monkeypatch):
    monkeypatch.delenv("ONLINE_MEMORY", raising=False)
    import tool_trace_rag.config as config

    reloaded = importlib.reload(config)

    assert reloaded.ONLINE_MEMORY is False


def test_online_memory_config_parses_true(monkeypatch):
    monkeypatch.setenv("ONLINE_MEMORY", "true")
    import tool_trace_rag.config as config

    reloaded = importlib.reload(config)

    assert reloaded.ONLINE_MEMORY is True
