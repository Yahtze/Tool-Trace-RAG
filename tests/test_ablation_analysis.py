from __future__ import annotations

from pathlib import Path

from analysis.scripts.ablation_analysis import compare_experiments
from analysis.scripts.common import write_json, write_jsonl


def make_experiment(root: Path, name: str, top_k: int, success_delta: float) -> Path:
    experiment = root / name
    write_json(experiment / "experiment_config.json", {"experiment_id": name, "top_k": top_k, "memory_filter": "successful_only"})
    write_json(
        experiment / "summary.json",
        {
            "experiment_id": name,
            "baseline": {"metrics": {"success_rate": 0.5, "avg_tool_calls": 2.0}},
            "retrieval": {"metrics": {"success_rate": 0.5 + success_delta, "avg_tool_calls": 1.5}},
            "comparison": {"success_rate_delta": success_delta, "avg_tool_calls_delta": -0.5},
        },
    )
    write_jsonl(experiment / "paired_results.jsonl", [])
    return experiment


def test_compare_experiments_preserves_config_and_metrics(tmp_path: Path):
    exp1 = make_experiment(tmp_path / "runs" / "experiments", "topk-1", 1, 0.0)
    exp3 = make_experiment(tmp_path / "runs" / "experiments", "topk-3", 3, 0.25)
    output = tmp_path / "analysis" / "artifacts" / "ablation"

    result = compare_experiments([exp1, exp3], output, analysis_root=tmp_path / "analysis")

    assert result["best_by_success_delta"] == "topk-3"
    assert result["runs"][0]["experiment_id"] == "topk-1"
    assert result["runs"][0]["top_k"] == 1
    assert result["runs"][1]["success_rate_delta"] == 0.25
    assert (output / "aggregate_summary.json").exists()
    assert (output / "ablation_results.csv").exists()
    assert (output / "report.md").exists()
