# Setup and Usage Guide

## 1) Prerequisites

- macOS, Linux, or WSL
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)

Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2) Clone and install

```bash
git clone <your-repo-url> tool-trace-RAG
cd tool-trace-RAG
uv sync --extra dev
```

## 3) Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set provider values (at minimum):
- `AGENT_API_KEY`
- `AGENT_MODEL`
- `AGENT_BASE_URL` (OpenAI-compatible)

Notes:
- `tool_trace_rag/config.py` auto-loads `.env`.
- `.env` values override conflicting shell values.

## 4) Validate install

### Local/test-safe path (no external provider calls)

```bash
uv run pytest
```

### Real runtime path (uses configured provider)

```bash
uv run python scripts/run_task.py "Where is order ord_1003?"
```

---

## CLI workflows

### A) Single task execution (`scripts/run_task.py`)

Basic:

```bash
uv run python scripts/run_task.py "Maya Chen asks if she can return the headphones from her last delivered order."
```

Save trace:

```bash
uv run python scripts/run_task.py \
  "Where is order ord_1003?" \
  --save-trace \
  --trace-dir runs/traces/local
```

Retrieval-only memory:

```bash
uv run python scripts/run_task.py \
  "Can I return the headphones from my last delivered order?" \
  --use-memory \
  --trace-dir runs/traces/local \
  --vector-dir runs/vector_store/local \
  --top-k 3 \
  --memory-filter successful_only
```

Online memory lifecycle (retrieve + run + persist + upsert):

```bash
uv run python scripts/run_task.py \
  "Can I return the headphones from my last delivered order?" \
  --online-memory \
  --trace-dir runs/traces/online \
  --vector-dir runs/vector_store/online \
  --top-k 3
```

Useful flags:
- `--data PATH`
- `--max-tool-calls INT`
- `--save-trace`
- `--trace-dir PATH`
- `--use-memory`
- `--online-memory`
- `--vector-dir PATH`
- `--collection NAME`
- `--top-k INT`
- `--memory-filter {successful_only,failed_only,all}`
- `--memory-strict`

### B) Index and query memory

Index persisted traces:

```bash
uv run python scripts/index_traces.py \
  --trace-dir runs/traces/local \
  --vector-dir runs/vector_store/local
```

Query memory index:

```bash
uv run python scripts/query_traces.py \
  "Customer asks for refund eligibility on delivered headphones" \
  --top-k 5
```

### C) Run evaluation (`scripts/run_eval.py`)

Default:

```bash
uv run python scripts/run_eval.py
```

With retrieval memory:

```bash
uv run python scripts/run_eval.py --use-memory --top-k 3
```

Persist one trace per task:

```bash
uv run python scripts/run_eval.py \
  --save-traces \
  --trace-dir runs/traces/eval
```

### D) Run controlled baseline-vs-retrieval experiment

```bash
uv run python scripts/run_experiment.py \
  --output-dir runs/experiments/smoke \
  --memory-trace-dir runs/traces/local \
  --memory-vector-dir runs/vector_store/local \
  --top-k 3 \
  --memory-filter successful_only
```

Writes:
- `experiment_config.json`
- `summary.json`
- `paired_results.jsonl`

### E) Run sequential online-memory study

```bash
uv run python scripts/run_sequential_study.py \
  --output-dir runs/sequential/default-sequence \
  --ordering original \
  --passes 1
```

---

## Offline analysis workflows (under `analysis/`)

Analyze one saved experiment:

```bash
uv run python analysis/scripts/analyze_experiment.py \
  --experiment-dir runs/experiments/smoke \
  --output-dir analysis/artifacts/experiment-smoke
```

Analyze one sequential study:

```bash
uv run python analysis/scripts/analyze_learning_curve.py \
  --sequence-dir runs/sequential/default-sequence \
  --output-dir analysis/artifacts/learning-curve
```

Compare multiple experiments (ablations):

```bash
uv run python analysis/scripts/compare_ablations.py \
  --experiments runs/experiments/topk-1 runs/experiments/topk-3 runs/experiments/topk-5 \
  --output-dir analysis/artifacts/topk-ablation
```

---

## Key environment variables

- Provider/runtime: `AGENT_BASE_URL`, `AGENT_API_KEY`, `AGENT_MODEL`, `AGENT_TIMEOUT_SECONDS`
- Agent controls: `AGENT_MAX_TOOL_CALLS`
- Paths: `CUSTOMER_SUPPORT_DATA_PATH`, `EVAL_TASKS_PATH`, `TRACE_DIR`, `VECTOR_DIR`, `VECTOR_COLLECTION_NAME`
- Memory: `USE_MEMORY`, `ONLINE_MEMORY`, `MEMORY_TOP_K`, `MEMORY_FILTER`, `MEMORY_SNIPPET_MAX_CHARS`, `MEMORY_STRICT`
- Embeddings: `EMBEDDING_MODEL`

## Repro checklist (new machine)

1. Install Python 3.11+ and `uv`.
2. Clone repo and run `uv sync --extra dev`.
3. Copy `.env.example` to `.env` and fill provider settings.
4. Run `uv run pytest`.
5. Run one real task via `scripts/run_task.py`.
6. (Optional) Save traces, index them, run with memory.
7. (Optional) Run experiment + analysis CLIs.