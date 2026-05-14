from __future__ import annotations

from pathlib import Path

from analysis.scripts.common import write_json, write_jsonl
from analysis.scripts.experiment_analysis import analyze_experiment_dir, categorize_pair, cluster_failures


def test_categorize_pair_names_required_outcomes():
    assert categorize_pair({"baseline_passed": False, "retrieval_passed": True, "tool_call_delta": -1}) == "retrieval_win"
    assert categorize_pair({"baseline_passed": True, "retrieval_passed": False, "tool_call_delta": 1}) == "retrieval_regression"
    assert categorize_pair({"baseline_passed": True, "retrieval_passed": True, "tool_call_delta": -1}) == "both_passed_fewer_tools"
    assert categorize_pair({"baseline_passed": True, "retrieval_passed": True, "tool_call_delta": 2}) == "both_passed_more_tools"
    assert categorize_pair({"baseline_passed": True, "retrieval_passed": True, "tool_call_delta": 0}) == "both_passed_same_tools"
    assert categorize_pair({"baseline_passed": False, "retrieval_passed": False, "tool_call_delta": 0}) == "both_failed"


def test_cluster_failures_counts_observable_reasons_and_tool_paths():
    rows = [
        {
            "task_id": "a",
            "baseline_passed": False,
            "retrieval_passed": True,
            "baseline_reasons": ["missing expected tool call: find_customer"],
            "retrieval_reasons": [],
            "baseline_tool_calls": 0,
            "retrieval_tool_calls": 2,
            "retrieval_metadata": {"injected_count": 2},
        },
        {
            "task_id": "b",
            "baseline_passed": True,
            "retrieval_passed": False,
            "baseline_reasons": [],
            "retrieval_reasons": ["answer missing expected text: refund"],
            "baseline_tool_calls": 1,
            "retrieval_tool_calls": 1,
            "retrieval_metadata": {"injected_count": 1},
        },
    ]

    clusters = cluster_failures(rows)

    assert clusters["reason_counts"]["missing expected tool call: find_customer"] == 1
    assert clusters["reason_counts"]["answer missing expected text: refund"] == 1
    assert clusters["category_counts"]["retrieval_win"] == 1
    assert clusters["category_counts"]["retrieval_regression"] == 1
    assert clusters["retrieval_count_distribution"] == {"1": 1, "2": 1}


def test_analyze_experiment_dir_writes_summary_and_per_task_outputs(tmp_path: Path):
    experiment = tmp_path / "runs" / "experiments" / "exp"
    output = tmp_path / "analysis" / "artifacts" / "exp-analysis"
    write_json(experiment / "experiment_config.json", {"experiment_id": "exp", "top_k": 3, "memory_filter": "successful_only"})
    write_json(
        experiment / "summary.json",
        {
            "experiment_id": "exp",
            "baseline": {"metrics": {"success_rate": 0.5, "avg_tool_calls": 1.0}},
            "retrieval": {"metrics": {"success_rate": 1.0, "avg_tool_calls": 0.5}},
            "comparison": {"success_rate_delta": 0.5, "avg_tool_calls_delta": -0.5},
        },
    )
    write_jsonl(
        experiment / "paired_results.jsonl",
        [
            {
                "task_id": "a",
                "prompt": "A?",
                "baseline_passed": False,
                "retrieval_passed": True,
                "baseline_reasons": ["missing expected tool call: find_customer"],
                "retrieval_reasons": [],
                "baseline_tool_calls": 0,
                "retrieval_tool_calls": 1,
                "tool_call_delta": 1,
                "retrieval_metadata": {"injected_count": 2},
            }
        ],
    )

    result = analyze_experiment_dir(experiment, output, analysis_root=tmp_path / "analysis")

    assert result["experiment_id"] == "exp"
    assert result["category_counts"] == {"retrieval_win": 1}
    assert (output / "aggregate_summary.json").exists()
    assert (output / "per_task_analysis.jsonl").exists()
    assert (output / "failure_clusters.json").exists()
    assert (output / "report.md").exists()
