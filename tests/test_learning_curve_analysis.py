from __future__ import annotations

from pathlib import Path

from analysis.scripts.common import write_json, write_jsonl
from analysis.scripts.learning_curve import analyze_sequence_dir, compute_learning_curve, repeated_pass_changes


def test_compute_learning_curve_adds_cumulative_and_rolling_metrics():
    steps = [
        {"step": 1, "pass_index": 1, "task_id": "a", "passed": True, "tool_calls": 1, "over_tooling": False, "under_tooling": False, "corpus_size_before": 0, "corpus_size_after": 1, "retrieved_memory_count": 0},
        {"step": 2, "pass_index": 1, "task_id": "b", "passed": False, "tool_calls": 3, "over_tooling": True, "under_tooling": False, "corpus_size_before": 1, "corpus_size_after": 2, "retrieved_memory_count": 1},
        {"step": 3, "pass_index": 1, "task_id": "c", "passed": True, "tool_calls": 2, "over_tooling": False, "under_tooling": False, "corpus_size_before": 2, "corpus_size_after": 3, "retrieved_memory_count": 2},
    ]

    curve = compute_learning_curve(steps, window=2)

    assert curve[0]["cumulative_success_rate"] == 1.0
    assert curve[1]["cumulative_success_rate"] == 0.5
    assert curve[2]["cumulative_success_rate"] == 0.6667
    assert curve[1]["rolling_success_rate"] == 0.5
    assert curve[2]["rolling_avg_tool_calls"] == 2.5
    assert curve[2]["cumulative_over_tooling_rate"] == 0.3333


def test_repeated_pass_changes_identifies_improvements_and_regressions():
    curve = [
        {"pass_index": 1, "task_id": "a", "passed": False, "tool_calls": 3},
        {"pass_index": 2, "task_id": "a", "passed": True, "tool_calls": 2},
        {"pass_index": 1, "task_id": "b", "passed": True, "tool_calls": 1},
        {"pass_index": 2, "task_id": "b", "passed": False, "tool_calls": 4},
    ]

    changes = repeated_pass_changes(curve)

    assert changes == [
        {"task_id": "a", "first_pass_passed": False, "last_pass_passed": True, "change": "improved", "tool_call_delta": -1},
        {"task_id": "b", "first_pass_passed": True, "last_pass_passed": False, "change": "regressed", "tool_call_delta": 3},
    ]


def test_analyze_sequence_dir_writes_learning_curve_outputs(tmp_path: Path):
    sequence = tmp_path / "runs" / "sequential" / "seq"
    output = tmp_path / "analysis" / "artifacts" / "seq-analysis"
    write_json(sequence / "sequence_config.json", {"sequence_id": "seq", "ordering": "original", "passes": 1})
    write_jsonl(sequence / "steps.jsonl", [{"step": 1, "pass_index": 1, "task_id": "a", "passed": True, "tool_calls": 1, "over_tooling": False, "under_tooling": False, "corpus_size_before": 0, "corpus_size_after": 1, "retrieved_memory_count": 0}])

    result = analyze_sequence_dir(sequence, output, analysis_root=tmp_path / "analysis", window=2)

    assert result["sequence_id"] == "seq"
    assert result["total_steps"] == 1
    assert (output / "aggregate_summary.json").exists()
    assert (output / "learning_curve.csv").exists()
    assert (output / "repeated_pass_changes.json").exists()
    assert (output / "report.md").exists()
