from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Prefer project-local .env values over ambient shell variables so local
        # runs are reproducible from the checked-out project configuration.
        os.environ[key] = value


_load_dotenv_file(".env")


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc


def _get_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean.")


AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.openai.com/v1")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY")
AGENT_MODEL = os.environ.get("AGENT_MODEL")
AGENT_TIMEOUT_SECONDS = _get_float("AGENT_TIMEOUT_SECONDS", 60.0)
AGENT_MAX_TOOL_CALLS = _get_int("AGENT_MAX_TOOL_CALLS", 8)

TRACE_DIR = Path(os.environ.get("TRACE_DIR", "runs/traces"))
VECTOR_DIR = Path(os.environ.get("VECTOR_DIR", "runs/vector_store"))
VECTOR_COLLECTION_NAME = os.environ.get("VECTOR_COLLECTION_NAME", "tool_trace_memory")
QUERY_TOP_K = _get_int("QUERY_TOP_K", 5)

USE_MEMORY = _get_bool("USE_MEMORY", False)
MEMORY_TOP_K = _get_int("MEMORY_TOP_K", 3)
MEMORY_FILTER = os.environ.get("MEMORY_FILTER", "successful_only")
MEMORY_SNIPPET_MAX_CHARS = _get_int("MEMORY_SNIPPET_MAX_CHARS", 1200)
MEMORY_STRICT = _get_bool("MEMORY_STRICT", False)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CUSTOMER_SUPPORT_DATA_PATH = os.environ.get("CUSTOMER_SUPPORT_DATA_PATH", "data/mock_customer_support.json")
EVAL_TASKS_PATH = os.environ.get("EVAL_TASKS_PATH", "data/eval_tasks_milestone_02.json")
