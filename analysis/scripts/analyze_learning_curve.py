#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.scripts.learning_curve import analyze_sequence_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a saved sequential online-memory study.")
    parser.add_argument("--sequence-dir", required=True, help="Path to runs/sequential/<sequence_id>.")
    parser.add_argument("--output-dir", required=True, help="Analysis output directory under analysis/.")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size for learning-curve metrics.")
    parser.add_argument("--analysis-root", default="analysis", help="Root analysis directory used for output-boundary validation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        analyze_sequence_dir(args.sequence_dir, args.output_dir, analysis_root=args.analysis_root, window=args.window)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    output = Path(args.output_dir)
    print(f"Learning-curve analysis written: {output}")
    print(f"- {output / 'aggregate_summary.json'}")
    print(f"- {output / 'learning_curve.csv'}")
    print(f"- {output / 'repeated_pass_changes.json'}")
    print(f"- {output / 'report.md'}")


if __name__ == "__main__":
    main()
