from __future__ import annotations

from pathlib import Path

from analysis.scripts.plots import write_bar_plot, write_line_plot


def test_write_line_plot_creates_png_or_text_fallback(tmp_path: Path):
    output = write_line_plot(
        tmp_path / "plots" / "learning_curve_success_rate.png",
        x=[1, 2, 3],
        y=[1.0, 0.5, 0.75],
        title="Success Rate",
        xlabel="step",
        ylabel="rate",
    )

    assert output.exists()
    assert output.suffix in {".png", ".txt"}


def test_write_bar_plot_creates_png_or_text_fallback(tmp_path: Path):
    output = write_bar_plot(
        tmp_path / "plots" / "ablation_success_rate.png",
        labels=["topk-1", "topk-3"],
        values=[0.0, 0.25],
        title="Ablation",
        ylabel="success delta",
    )

    assert output.exists()
    assert output.suffix in {".png", ".txt"}
