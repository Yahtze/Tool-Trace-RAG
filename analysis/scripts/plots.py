from __future__ import annotations

from pathlib import Path
from typing import Sequence


def write_line_plot(path: str | Path, x: Sequence[object], y: Sequence[float], title: str, xlabel: str, ylabel: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return _write_text_fallback(target, title, xlabel, ylabel, list(zip(x, y)))
    plt.figure()
    plt.plot(list(x), list(y), marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def write_bar_plot(path: str | Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return _write_text_fallback(target, title, "label", ylabel, list(zip(labels, values)))
    plt.figure()
    plt.bar(list(labels), list(values))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target


def _write_text_fallback(path: Path, title: str, xlabel: str, ylabel: str, rows: list[tuple[object, object]]) -> Path:
    fallback = path.with_suffix(".txt")
    lines = [title, f"{xlabel},{ylabel}"]
    lines.extend(f"{left},{right}" for left, right in rows)
    fallback.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fallback
