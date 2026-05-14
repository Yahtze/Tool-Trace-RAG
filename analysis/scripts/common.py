from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_analysis_output_dir(output_dir: str | Path, analysis_root: str | Path = "analysis") -> Path:
    output = Path(output_dir).resolve()
    root = Path(analysis_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"analysis output must be under {root}: {output}") from exc
    output.mkdir(parents=True, exist_ok=True)
    return output


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def require_files(paths: Iterable[str | Path]) -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required artifact(s): " + ", ".join(missing))
