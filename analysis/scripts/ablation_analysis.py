from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from analysis.scripts.common import ensure_analysis_output_dir, read_json, require_files, write_json
from analysis.scripts.plots import write_bar_plot


def compare_experiments(experiment_dirs: list[str | Path], output_dir: str | Path, analysis_root: str | Path = "analysis") -> dict[str, Any]:
    output = ensure_analysis_output_dir(output_dir, analysis_root=analysis_root)
    rows = [_load_experiment_row(Path(path)) for path in experiment_dirs]
    best = max(rows, key=lambda row: float(row.get("success_rate_delta", 0.0)))["experiment_id"] if rows else None
    summary = {
        "analysis_type": "ablation_comparison",
        "experiment_count": len(rows),
        "best_by_success_delta": best,
        "runs": rows,
    }
    write_json(output / "analysis_config.json", {"input_experiment_dirs": [str(path) for path in experiment_dirs], "analysis_type": "ablation_comparison"})
    write_json(output / "aggregate_summary.json", summary)
    write_csv(output / "ablation_results.csv", rows)
    write_bar_plot(
        output / "plots" / "ablation_success_rate.png",
        labels=[str(row["experiment_id"]) for row in rows],
        values=[float(row.get("success_rate_delta", 0.0)) for row in rows],
        title="Success Rate Delta By Ablation",
        ylabel="success_rate_delta",
    )
    write_report(output / "report.md", summary)
    return summary


def _load_experiment_row(experiment: Path) -> dict[str, Any]:
    require_files([experiment / "experiment_config.json", experiment / "summary.json"])
    config = read_json(experiment / "experiment_config.json")
    summary = read_json(experiment / "summary.json")
    comparison = summary.get("comparison", {})
    return {
        "experiment_id": str(summary.get("experiment_id") or config.get("experiment_id") or experiment.name),
        "experiment_dir": str(experiment),
        "top_k": config.get("top_k"),
        "memory_filter": config.get("memory_filter"),
        "success_rate_delta": comparison.get("success_rate_delta", 0.0),
        "avg_tool_calls_delta": comparison.get("avg_tool_calls_delta", 0.0),
        "baseline_success_rate": summary.get("baseline", {}).get("metrics", {}).get("success_rate"),
        "retrieval_success_rate": summary.get("retrieval", {}).get("metrics", {}).get("success_rate"),
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment_id", "top_k", "memory_filter", "success_rate_delta", "avg_tool_calls_delta", "baseline_success_rate", "retrieval_success_rate", "experiment_dir"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: str | Path, summary: dict[str, Any]) -> None:
    lines = ["# Ablation Comparison", "", f"- Experiment count: {summary['experiment_count']}", f"- Best by success delta: {summary['best_by_success_delta']}", "", "## Runs", ""]
    for row in summary.get("runs", []):
        lines.append(f"- {row['experiment_id']}: success_delta={row['success_rate_delta']}, top_k={row.get('top_k')}, filter={row.get('memory_filter')}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
