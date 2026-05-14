from __future__ import annotations

from pathlib import Path

import pytest

from analysis.scripts.common import ensure_analysis_output_dir, read_json, read_jsonl, write_json, write_jsonl


def test_ensure_analysis_output_dir_accepts_analysis_path(tmp_path: Path):
    output = tmp_path / "analysis" / "artifacts" / "unit"
    result = ensure_analysis_output_dir(output, analysis_root=tmp_path / "analysis")

    assert result == output
    assert output.exists()


def test_ensure_analysis_output_dir_rejects_non_analysis_path(tmp_path: Path):
    with pytest.raises(ValueError, match="analysis output must be under"):
        ensure_analysis_output_dir(tmp_path / "runs" / "bad", analysis_root=tmp_path / "analysis")


def test_json_and_jsonl_helpers_round_trip(tmp_path: Path):
    json_path = tmp_path / "payload.json"
    jsonl_path = tmp_path / "items.jsonl"

    write_json(json_path, {"name": "unit", "count": 2})
    write_jsonl(jsonl_path, [{"id": "a"}, {"id": "b"}])

    assert read_json(json_path) == {"name": "unit", "count": 2}
    assert read_jsonl(jsonl_path) == [{"id": "a"}, {"id": "b"}]
