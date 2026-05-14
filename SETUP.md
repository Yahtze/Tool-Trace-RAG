# Setup and Usage Guide

## 1) Prerequisites
- macOS/Linux/WSL
- Python 3.11+
- `uv` installed

Install `uv` (if needed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2) Clone and enter the project
```bash
git clone <your-repo-url> tool-trace-RAG
cd tool-trace-RAG
```

## 3) Create environment and install dependencies (with uv)
```bash
uv sync
```

For dev/test tools (pytest extras):
```bash
uv sync --extra dev
```

## 4) Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and set at least:
- `AGENT_API_KEY`
- `AGENT_MODEL`
- `AGENT_BASE_URL` (keep default if using OpenAI)

Notes:
- `tool_trace_rag/config.py` auto-loads `.env` at runtime.
- `.env` values are preferred over conflicting shell env values.

## 5) Validate installation
Run tests:
```bash
uv run pytest
```

## 6) Run one task (`scripts/run_task.py`)
Basic:
```bash
uv run python scripts/run_task.py "Maya Chen asks if she can return the headphones from her last delivered order."
```

### `run_task.py` arguments

**Positional**
- `task` (required): task text for the agent.

**Flags**
- `--data PATH`  
  Path to mock customer support data. Default: `CUSTOMER_SUPPORT_DATA_PATH` from `.env`.
- `--max-tool-calls INT`  
  Maximum tool calls before aborting. Default: `AGENT_MAX_TOOL_CALLS`.
- `--save-trace`  
  Persist run trace JSON.
- `--trace-dir PATH`  
  Trace output directory. Implies save trace.
- `--use-memory`  
  Enable retrieval + prompt memory injection.
- `--online-memory`  
  Run single-task online lifecycle: retrieve, execute, persist trace, embed, and upsert.
- `--vector-dir PATH`  
  Local vector store directory. Default: `VECTOR_DIR`.
- `--collection NAME`  
  Vector collection name. Default: `VECTOR_COLLECTION_NAME`.
- `--top-k INT`  
  Number of retrieved memories. Default: `MEMORY_TOP_K`.
- `--memory-filter {successful_only,failed_only,all}`  
  Retrieval filter policy. Default: `MEMORY_FILTER`.
- `--memory-strict`  
  Fail run on retrieval errors (otherwise fallback to no-memory).

Example with trace persistence:
```bash
uv run python scripts/run_task.py "Where is order ord_1003?" --save-trace --trace-dir runs/traces/local
```

Example with memory enabled:
```bash
uv run python scripts/run_task.py "Can I return the headphones from my last delivered order?" \
  --use-memory \
  --vector-dir runs/vector_store \
  --trace-dir runs/traces \
  --top-k 3 \
  --memory-filter successful_only
```

Example with online memory enabled:
```bash
uv run python scripts/run_task.py "Can I return the headphones from my last delivered order?" \
  --online-memory \
  --use-memory \
  --vector-dir runs/vector_store \
  --trace-dir runs/traces \
  --top-k 3
```

## 7) Build and query memory index (optional but recommended for `--use-memory`)
Index traces:
```bash
uv run python scripts/index_traces.py --trace-dir runs/traces --vector-dir runs/vector_store
```

Query index only:
```bash
uv run python scripts/query_traces.py "Customer asks for refund eligibility on delivered headphones" --top-k 5
```

## 8) Run evaluation (optional)
Without memory:
```bash
uv run python scripts/run_eval.py
```

With memory:
```bash
uv run python scripts/run_eval.py --use-memory --top-k 3
```

Persist one trace per eval task:
```bash
uv run python scripts/run_eval.py --save-traces --trace-dir runs/traces/eval-baseline
```

## 9) Key `.env` settings
- Provider: `AGENT_BASE_URL`, `AGENT_API_KEY`, `AGENT_MODEL`, `AGENT_TIMEOUT_SECONDS`
- Agent: `AGENT_MAX_TOOL_CALLS`
- Paths: `CUSTOMER_SUPPORT_DATA_PATH`, `EVAL_TASKS_PATH`, `TRACE_DIR`, `VECTOR_DIR`, `VECTOR_COLLECTION_NAME`
- Memory: `USE_MEMORY`, `ONLINE_MEMORY`, `MEMORY_TOP_K`, `MEMORY_FILTER`, `MEMORY_SNIPPET_MAX_CHARS`, `MEMORY_STRICT`
- Embeddings: `EMBEDDING_MODEL`

## 10) Recreate on another machine checklist
1. Install Python 3.11+ and `uv`.
2. Clone repo.
3. Run `uv sync --extra dev`.
4. Copy `.env.example` to `.env` and fill provider credentials.
5. Run `uv run pytest`.
6. Run `uv run python scripts/run_task.py "<task>"`.
7. (Optional) Save traces, index them, and re-run with `--use-memory`.
