from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from analysis.scripts.common import ensure_analysis_output_dir, read_json, read_jsonl, require_files, write_json, write_jsonl


def categorize_pair(row: dict[str, Any]) -> str:
    baseline_passed = bool(row.get("baseline_passed"))
    retrieval_passed = bool(row.get("retrieval_passed"))
    tool_delta = int(row.get("tool_call_delta", 0))
    if retrieval_passed and not baseline_passed:
        return "retrieval_win"
    if baseline_passed and not retrieval_passed:
        return "retrieval_regression"
    if baseline_passed and retrieval_passed:
        if tool_delta < 0:
            return "both_passed_fewer_tools"
        if tool_delta > 0:
            return "both_passed_more_tools"
        return "both_passed_same_tools"
    return "both_failed"


def retrieval_count(row: dict[str, Any]) -> int:
    metadata = row.get("retrieval_metadata") or {}
    if isinstance(metadata, dict):
        if "injected_count" in metadata:
            return int(metadata["injected_count"] or 0)
        memories = metadata.get("memories")
        if isinstance(memories, list):
            return len(memories)
    return 0


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["category"] = categorize_pair(item)
        item["retrieved_memory_count"] = retrieval_count(item)
        enriched.append(item)
    return enriched


def cluster_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    retrieval_counts: Counter[str] = Counter()
    for row in rows:
        category_counts[categorize_pair(row)] += 1
        retrieval_counts[str(retrieval_count(row))] += 1
        for reason in row.get("baseline_reasons", []) or []:
            reason_counts[str(reason)] += 1
        for reason in row.get("retrieval_reasons", []) or []:
            reason_counts[str(reason)] += 1
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "retrieval_count_distribution": dict(sorted(retrieval_counts.items(), key=lambda item: int(item[0]))),
    }


def analyze_experiment_dir(experiment_dir: str | Path, output_dir: str | Path, analysis_root: str | Path = "analysis") -> dict[str, Any]:
    experiment = Path(experiment_dir)
    require_files([
        experiment / "experiment_config.json",
        experiment / "summary.json",
        experiment / "paired_results.jsonl",
    ])
    output = ensure_analysis_output_dir(output_dir, analysis_root=analysis_root)
    config = read_json(experiment / "experiment_config.json")
    summary = read_json(experiment / "summary.json")
    rows = enrich_rows(read_jsonl(experiment / "paired_results.jsonl"))
    clusters = cluster_failures(rows)
    aggregate = {
        "experiment_id": summary.get("experiment_id") or config.get("experiment_id") or experiment.name,
        "experiment_dir": str(experiment),
        "config": config,
        "baseline_metrics": summary.get("baseline", {}).get("metrics", {}),
        "retrieval_metrics": summary.get("retrieval", {}).get("metrics", {}),
        "comparison": summary.get("comparison", {}),
        "category_counts": clusters["category_counts"],
        "total_tasks": len(rows),
    }
    write_json(output / "analysis_config.json", {"input_experiment_dir": str(experiment), "analysis_type": "experiment"})
    write_json(output / "aggregate_summary.json", aggregate)
    write_jsonl(output / "per_task_analysis.jsonl", rows)
    write_json(output / "failure_clusters.json", clusters)
    write_report(output / "report.md", aggregate, clusters)
    return aggregate


def write_report(path: str | Path, aggregate: dict[str, Any], clusters: dict[str, Any]) -> None:
    comparison = aggregate.get("comparison", {})
    lines = [
        f"# Experiment Analysis: {aggregate['experiment_id']}",
        "",
        "## Observed deltas",
        "",
        f"- Success rate delta: {comparison.get('success_rate_delta', 'n/a')}",
        f"- Average tool calls delta: {comparison.get('avg_tool_calls_delta', 'n/a')}",
        "",
        "## Outcome categories",
        "",
    ]
    for name, count in aggregate.get("category_counts", {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Common failure reasons", ""])
    for reason, count in clusters.get("reason_counts", {}).items():
        lines.append(f"- {reason}: {count}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
