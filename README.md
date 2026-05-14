# tool-trace-RAG

Trace-based memory for tool-calling agents.

## What this project does

`tool-trace-RAG` runs an OpenAI-compatible tool-calling agent in a deterministic customer-support domain, records full run traces, and reuses prior traces as retrieval memory to improve later tool-use behavior.

Core capabilities:
- Tool-calling runtime with deterministic domain tools.
- Evaluation harness for task success and tool-use efficiency.
- Trace persistence to local JSON artifacts.
- Local embedding + vector retrieval over traces.
- Retrieval-time memory injection.
- Online memory lifecycle (retrieve → run → persist → upsert).
- Controlled baseline-vs-retrieval experiments.
- Offline analysis layer under `analysis/` for experiments, learning curves, ablations, and plots.

## Repository layout

- `tool_trace_rag/` — runtime, evaluation, memory, traces, experiments.
- `scripts/` — CLI entrypoints.
- `analysis/` — analysis code + derived artifacts.
- `data/` — deterministic domain data + eval tasks.
- `runs/` — runtime outputs (traces, vector stores, experiment/sequential artifacts).
- `tests/` — unit/integration tests.

## Quickstart

See full setup in [`SETUP.md`](./SETUP.md).

```bash
uv sync --extra dev
cp .env.example .env
uv run pytest
```

## Common workflows

### 1) Run one task (real provider)

```bash
uv run python scripts/run_task.py \
  "Maya Chen asks if she can return the headphones from her last delivered order."
```

### 2) Save traces + build memory corpus

```bash
uv run python scripts/run_task.py \
  "Where is order ord_1003?" \
  --save-trace \
  --trace-dir runs/traces/local

uv run python scripts/index_traces.py \
  --trace-dir runs/traces/local \
  --vector-dir runs/vector_store/local
```

### 3) Retrieval-only memory run

```bash
uv run python scripts/run_task.py \
  "Can I return the headphones from my last delivered order?" \
  --use-memory \
  --trace-dir runs/traces/local \
  --vector-dir runs/vector_store/local \
  --top-k 3 \
  --memory-filter successful_only
```

### 4) Online memory run (retrieve + update corpus)

```bash
uv run python scripts/run_task.py \
  "Can I return the headphones from my last delivered order?" \
  --online-memory \
  --trace-dir runs/traces/online \
  --vector-dir runs/vector_store/online \
  --top-k 3
```

### 5) Run controlled experiment

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

### 6) Run sequential online-memory study

```bash
uv run python scripts/run_sequential_study.py \
  --output-dir runs/sequential/default-sequence \
  --ordering original \
  --passes 1
```

### 7) Analyze saved artifacts (offline, no live LLM calls)

```bash
uv run python analysis/scripts/analyze_experiment.py \
  --experiment-dir runs/experiments/smoke \
  --output-dir analysis/artifacts/experiment-smoke

uv run python analysis/scripts/analyze_learning_curve.py \
  --sequence-dir runs/sequential/default-sequence \
  --output-dir analysis/artifacts/learning-curve

uv run python analysis/scripts/compare_ablations.py \
  --experiments runs/experiments/topk-1 runs/experiments/topk-3 runs/experiments/topk-5 \
  --output-dir analysis/artifacts/topk-ablation
```

## Local/test-safe workflows (no provider calls)

For deterministic validation without external API usage, use tests:

```bash
uv run pytest
```

The test suite uses fakes/mocks for provider-dependent paths where appropriate.

## Notes

- Analysis outputs must be under `analysis/`.
- Sequential runtime artifacts go under `runs/sequential/`.
- Retrieval-only experiment arm does not mutate memory corpus.
- Online memory mode does mutate corpus (persist + upsert).