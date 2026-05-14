from __future__ import annotations

import subprocess
import sys

from tool_trace_rag.eval.schema import EvalMetrics
from tool_trace_rag.experiments.schema import ArmSummary, ExperimentComparison, ExperimentResult


def test_run_experiment_help_includes_memory_and_output_flags():
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dataset" in result.stdout
    assert "--memory-trace-dir" in result.stdout
    assert "--memory-vector-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--top-k" in result.stdout
    assert "--memory-filter" in result.stdout


def test_run_experiment_main_prints_summary_without_live_provider(monkeypatch, capsys, tmp_path):
    import scripts.run_experiment as run_experiment

    monkeypatch.setattr(run_experiment, "load_eval_tasks", lambda path: [object(), object()])

    class FakeRunner:
        def __init__(self, bootstrap):
            self.bootstrap = bootstrap

        def run(self, tasks, config):
            metrics = EvalMetrics(
                total_tasks=2,
                passed_tasks=1,
                failed_tasks=1,
                success_rate=0.5,
                tool_call_rate=0.5,
                avg_tool_calls=1.0,
                avg_tool_calls_when_tools_required=1.0,
                avg_tool_calls_when_tools_not_required=0.0,
                over_tooling_rate=0.0,
                under_tooling_rate=0.0,
                max_tool_call_violations=0,
                duplicate_tool_call_rate=0.0,
                error_rate=0.0,
            )
            return ExperimentResult(
                config=config,
                baseline=ArmSummary(name="baseline", metrics=metrics, trace_dir="baseline/traces"),
                retrieval=ArmSummary(name="retrieval", metrics=metrics, trace_dir="retrieval/traces"),
                comparison=ExperimentComparison(
                    success_rate_delta=0.0,
                    avg_tool_calls_delta=0.0,
                    over_tooling_rate_delta=0.0,
                    under_tooling_rate_delta=0.0,
                    retrieval_wins=0,
                    baseline_wins=0,
                    both_passed=1,
                    both_failed=1,
                    mean_tool_call_delta=0.0,
                ),
                paired_results=[],
            )

    monkeypatch.setattr(run_experiment, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(run_experiment, "RuntimeBootstrap", lambda: object())

    run_experiment.main(
        [
            "--dataset",
            "fake-dataset.json",
            "--output-dir",
            str(tmp_path / "exp-cli"),
            "--memory-trace-dir",
            str(tmp_path / "memory-traces"),
            "--memory-vector-dir",
            str(tmp_path / "memory-vectors"),
        ]
    )

    output = capsys.readouterr().out
    assert "Experiment: exp-cli" in output
    assert "Dataset tasks: 2" in output
    assert "Arms: baseline, retrieval" in output
    assert "Baseline success rate: 0.5" in output
    assert "Retrieval success rate: 0.5" in output
    assert "Artifacts written: " in output
