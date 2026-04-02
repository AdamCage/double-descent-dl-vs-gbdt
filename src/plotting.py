"""Plotly helper: save interactive HTML (fully offline) and static PNG."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


def save_fig(fig: go.Figure, path: Path, *, width: int = 1200, height: int = 700) -> None:
    """Save a Plotly figure as .html (self-contained) and .png side-by-side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    html_path = path.with_suffix(".html")
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
    )

    try:
        png_path = path.with_suffix(".png")
        pio.write_image(fig, str(png_path), width=width, height=height, scale=2)
    except Exception:
        pass
