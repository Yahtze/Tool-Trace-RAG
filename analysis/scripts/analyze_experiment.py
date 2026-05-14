#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.scripts.experiment_analysis import analyze_experiment_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a saved baseline-vs-retrieval experiment.")
    parser.add_argument("--experiment-dir", required=True, help="Path to runs/experiments/<experiment_id>.")
    parser.add_argument("--output-dir", required=True, help="Analysis output directory under analysis/.")
    parser.add_argument("--analysis-root", default="analysis", help="Root analysis directory used for output-boundary validation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        analyze_experiment_dir(args.experiment_dir, args.output_dir, analysis_root=args.analysis_root)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    output = Path(args.output_dir)
    print(f"Analysis written: {output}")
    print(f"- {output / 'aggregate_summary.json'}")
    print(f"- {output / 'per_task_analysis.jsonl'}")
    print(f"- {output / 'failure_clusters.json'}")
    print(f"- {output / 'report.md'}")


if __name__ == "__main__":
    main()
