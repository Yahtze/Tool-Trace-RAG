from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from analysis.scripts.common import write_json, write_jsonl


def test_analyze_experiment_help_lists_required_flags():
    result = subprocess.run(
        [sys.executable, "analysis/scripts/analyze_experiment.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--experiment-dir" in result.stdout
    assert "--output-dir" in result.stdout


def test_analyze_experiment_cli_writes_under_analysis(tmp_path: Path):
    experiment = tmp_path / "runs" / "experiments" / "exp"
    output = tmp_path / "analysis" / "artifacts" / "exp-analysis"
    write_json(experiment / "experiment_config.json", {"experiment_id": "exp"})
    write_json(experiment / "summary.json", {"experiment_id": "exp", "comparison": {"success_rate_delta": 0.0}})
    write_jsonl(experiment / "paired_results.jsonl", [])

    result = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/analyze_experiment.py",
            "--experiment-dir",
            str(experiment),
            "--output-dir",
            str(output),
            "--analysis-root",
            str(tmp_path / "analysis"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "aggregate_summary.json" in result.stdout
    assert (output / "aggregate_summary.json").exists()


def test_analyze_experiment_cli_rejects_runs_output(tmp_path: Path):
    experiment = tmp_path / "runs" / "experiments" / "exp"
    write_json(experiment / "experiment_config.json", {"experiment_id": "exp"})
    write_json(experiment / "summary.json", {"experiment_id": "exp"})
    write_jsonl(experiment / "paired_results.jsonl", [])

    result = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/analyze_experiment.py",
            "--experiment-dir",
            str(experiment),
            "--output-dir",
            str(tmp_path / "runs" / "bad"),
            "--analysis-root",
            str(tmp_path / "analysis"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "analysis output must be under" in result.stderr


def test_analyze_learning_curve_help_lists_required_flags():
    result = subprocess.run(
        [sys.executable, "analysis/scripts/analyze_learning_curve.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--sequence-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--window" in result.stdout


def test_compare_ablations_help_lists_required_flags():
    result = subprocess.run(
        [sys.executable, "analysis/scripts/compare_ablations.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--experiments" in result.stdout
    assert "--output-dir" in result.stdout
