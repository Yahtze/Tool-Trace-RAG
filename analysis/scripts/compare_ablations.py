#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.scripts.ablation_analysis import compare_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare saved retrieval ablation experiment artifacts.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment directories to compare.")
    parser.add_argument("--output-dir", required=True, help="Analysis output directory under analysis/.")
    parser.add_argument("--analysis-root", default="analysis", help="Root analysis directory used for output-boundary validation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        compare_experiments(args.experiments, args.output_dir, analysis_root=args.analysis_root)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    output = Path(args.output_dir)
    print(f"Ablation comparison written: {output}")
    print(f"- {output / 'aggregate_summary.json'}")
    print(f"- {output / 'ablation_results.csv'}")
    print(f"- {output / 'report.md'}")


if __name__ == "__main__":
    main()
