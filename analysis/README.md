# Analysis

This folder contains all milestone-08 analysis code and derived artifacts.

Runtime traces, experiment runs, and sequential study outputs may live under `runs/`. Anything that loads those outputs to inspect, aggregate, visualize, or report on them belongs under `analysis/`.

## Layout

- `scripts/`: importable analysis modules and CLI entrypoints.
- `notebooks/`: optional exploratory notebooks.
- `plots/`: generated standalone plots.
- `reports/`: generated standalone reports.
- `artifacts/`: analysis-run output directories.
- `fixtures/`: small deterministic analysis fixtures.

## Output rule

Analysis CLIs must write derived outputs under `analysis/`, usually `analysis/artifacts/<analysis_run_id>/`.

## Commands

Analyze a saved milestone-07 experiment:

```bash
uv run python analysis/scripts/analyze_experiment.py \
  --experiment-dir runs/experiments/milestone-07-smoke \
  --output-dir analysis/artifacts/milestone-08-smoke
```

Analyze a saved sequential online-memory study:

```bash
uv run python analysis/scripts/analyze_learning_curve.py \
  --sequence-dir runs/sequential/milestone-08-online \
  --output-dir analysis/artifacts/milestone-08-learning-curve
```

Compare retrieval ablation runs:

```bash
uv run python analysis/scripts/compare_ablations.py \
  --experiments runs/experiments/topk-1 runs/experiments/topk-3 runs/experiments/topk-5 \
  --output-dir analysis/artifacts/milestone-08-topk-ablation
```
