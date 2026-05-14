from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from analysis.scripts.common import ensure_analysis_output_dir, read_json, read_jsonl, require_files, write_json
from analysis.scripts.plots import write_line_plot


def _round(value: float) -> float:
    return round(value, 4)


def compute_learning_curve(steps: list[dict[str, Any]], window: int = 5) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    passed_count = 0
    over_count = 0
    under_count = 0
    for index, step in enumerate(steps, start=1):
        passed_count += 1 if step.get("passed") else 0
        over_count += 1 if step.get("over_tooling") else 0
        under_count += 1 if step.get("under_tooling") else 0
        recent = steps[max(0, index - window):index]
        recent_passes = sum(1 for item in recent if item.get("passed"))
        recent_tool_calls = sum(int(item.get("tool_calls", 0)) for item in recent)
        item = dict(step)
        item["cumulative_success_rate"] = _round(passed_count / index)
        item["rolling_success_rate"] = _round(recent_passes / len(recent)) if recent else 0.0
        item["rolling_avg_tool_calls"] = _round(recent_tool_calls / len(recent)) if recent else 0.0
        item["cumulative_over_tooling_rate"] = _round(over_count / index)
        item["cumulative_under_tooling_rate"] = _round(under_count / index)
        curve.append(item)
    return curve


def repeated_pass_changes(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in curve:
        by_task[str(row["task_id"])].append(row)
    changes: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id], key=lambda item: int(item.get("pass_index", 1)))
        if len(rows) < 2:
            continue
        first = rows[0]
        last = rows[-1]
        first_passed = bool(first.get("passed"))
        last_passed = bool(last.get("passed"))
        if not first_passed and last_passed:
            change = "improved"
        elif first_passed and not last_passed:
            change = "regressed"
        elif first_passed and last_passed:
            change = "stayed_passed"
        else:
            change = "stayed_failed"
        changes.append(
            {
                "task_id": task_id,
                "first_pass_passed": first_passed,
                "last_pass_passed": last_passed,
                "change": change,
                "tool_call_delta": int(last.get("tool_calls", 0)) - int(first.get("tool_calls", 0)),
            }
        )
    return changes


def analyze_sequence_dir(sequence_dir: str | Path, output_dir: str | Path, analysis_root: str | Path = "analysis", window: int = 5) -> dict[str, Any]:
    sequence = Path(sequence_dir)
    require_files([sequence / "sequence_config.json", sequence / "steps.jsonl"])
    output = ensure_analysis_output_dir(output_dir, analysis_root=analysis_root)
    config = read_json(sequence / "sequence_config.json")
    steps = read_jsonl(sequence / "steps.jsonl")
    curve = compute_learning_curve(steps, window=window)
    changes = repeated_pass_changes(curve)
    summary = {
        "sequence_id": config.get("sequence_id") or sequence.name,
        "sequence_dir": str(sequence),
        "config": config,
        "total_steps": len(curve),
        "final_success_rate": curve[-1]["cumulative_success_rate"] if curve else 0.0,
        "final_corpus_size": curve[-1].get("corpus_size_after", 0) if curve else 0,
        "change_counts": _count_changes(changes),
    }
    write_json(output / "analysis_config.json", {"input_sequence_dir": str(sequence), "analysis_type": "learning_curve", "window": window})
    write_json(output / "aggregate_summary.json", summary)
    write_csv(output / "learning_curve.csv", curve)
    plots_dir = output / "plots"
    write_line_plot(
        plots_dir / "learning_curve_success_rate.png",
        x=[row.get("step", index + 1) for index, row in enumerate(curve)],
        y=[float(row.get("cumulative_success_rate", 0.0)) for row in curve],
        title="Cumulative Success Rate",
        xlabel="step",
        ylabel="success_rate",
    )
    write_line_plot(
        plots_dir / "learning_curve_tool_calls.png",
        x=[row.get("step", index + 1) for index, row in enumerate(curve)],
        y=[float(row.get("rolling_avg_tool_calls", 0.0)) for row in curve],
        title="Rolling Average Tool Calls",
        xlabel="step",
        ylabel="tool_calls",
    )
    write_json(output / "repeated_pass_changes.json", {"changes": changes})
    write_report(output / "report.md", summary)
    return summary


def _count_changes(changes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in changes:
        key = str(item["change"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("\n", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Learning Curve Analysis: {summary['sequence_id']}",
        "",
        f"- Total steps: {summary['total_steps']}",
        f"- Final success rate: {summary['final_success_rate']}",
        f"- Final corpus size: {summary['final_corpus_size']}",
        "",
        "## Repeated-pass changes",
        "",
    ]
    for name, count in summary.get("change_counts", {}).items():
        lines.append(f"- {name}: {count}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
