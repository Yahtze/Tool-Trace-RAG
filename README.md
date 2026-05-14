# tool-trace-RAG

## Overview
`tool-trace-RAG` is a research-oriented agent project focused on **improving tool-use quality through trace-based retrieval**.

The system runs an OpenAI-compatible tool-calling agent in a deterministic customer-support domain, captures full observable run traces, and reuses relevant past traces as memory for future tasks.

## Goals
The core goals are to build a reliable loop that can:
1. Solve tasks with tool calls when needed.
2. Measure task success and tool-call efficiency.
3. Persist traces of real agent behavior.
4. Index those traces in a local vector store.
5. Retrieve relevant prior traces and inject them as compact prompt guidance.

### Motivation
Many agent systems underperform not only because of language-model limitations, but because they make inefficient tool decisions (wrong tool, missing tool, repeated tool calls, or unnecessary tool use). This project tests the hypothesis that **observable prior tool-use traces** can be reused as practical memory to guide better decisions on new but similar tasks.

The experiment is structured to separate concerns so results are interpretable:
- first prove the base tool-calling loop,
- then establish evaluation baselines,
- then add persistence,
- then add retrieval infrastructure,
- then inject retrieved examples into prompts.

This staged design helps answer a clear question: whether trace-based memory changes agent behavior in useful ways, without mixing in unrelated system changes.

## Current high-level architecture
- **Agent runtime**: OpenAI-compatible chat-completions tool-calling loop.
- **Domain tools**: deterministic customer/order/refund tools over mock data.
- **Evaluation harness**: dataset-driven scoring for success and tool efficiency.
- **Trace store**: local JSON trace persistence for every run.
- **Memory index**: local embedding + vector search over persisted traces.
- **Retrieval injector**: optional prompt-time memory context built from top-k similar traces.

## Scope boundaries
This repository focuses on local, reproducible infrastructure for trace capture, evaluation, indexing, retrieval injection, online memory updates, and controlled baseline-vs-retrieval experiments.

## Single-run online memory
Use `--online-memory` when a completed run should immediately update local memory:

```bash
uv run python scripts/run_task.py \
  "Maya Chen asks if she can return the headphones from her last delivered order." \
  --online-memory \
  --trace-dir runs/traces/online \
  --vector-dir runs/vector_store/online \
  --top-k 3
```

`--use-memory` stays retrieval-only. `--online-memory` retrieves memories, runs the agent, persists the new trace, embeds it, and upserts it into the configured vector store.

## Controlled memory experiments
Run a two-arm experiment comparing no-memory baseline behavior against retrieval-only memory:

```bash
uv run python scripts/run_experiment.py \
  --dataset data/eval_tasks_milestone_02.json \
  --memory-trace-dir runs/traces/memory \
  --memory-vector-dir runs/vector_store/memory \
  --output-dir runs/experiments/milestone-07-smoke \
  --top-k 3 \
  --memory-filter successful_only
```

The experiment runner writes `experiment_config.json`, `summary.json`, and `paired_results.jsonl`. The retrieval arm is read-only and does not update the memory corpus.

## Setup and usage
For full local setup instructions (using `uv`), environment configuration, and script usage details, see:

- [`SETUP.md`](./SETUP.md)
