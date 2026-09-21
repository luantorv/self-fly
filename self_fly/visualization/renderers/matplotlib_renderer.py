from __future__ import annotations

from matplotlib.patches import Polygon, Rectangle

from self_fly.stimuli.features import FeatureVector

from ..render import background_color, fill_color, is_dashed, movement_arrow, polygon_points


class MatplotlibRenderer:
    """Draws the same primitives render.py computes (geometry, fill color,
    background, movement arrow) onto a matplotlib Axes instead of a
    tk.Canvas -- same FeatureVector-only contract, no tkinter dependency,
    usable headless (e.g. batch snapshot exports, or embedding in the
    figure the GUI already has)."""

    def __init__(self, ax):
        self.ax = ax

    def render(self, features: FeatureVector, cx: float = 0.0, cy: float = 0.0, size: float = 1.0) -> None:
        self.ax.clear()
        self.ax.set_xlim(cx - size * 1.3, cx + size * 1.3)
        self.ax.set_ylim(cy - size * 1.3, cy + size * 1.3)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.ax.add_patch(
            Rectangle(
                (cx - size * 1.3, cy - size * 1.3),
                size * 2.6,
                size * 2.6,
                facecolor=background_color(features),
                edgecolor="none",
                zorder=0,
            )
        )

        raw_points = polygon_points(features, cx, cy, size)
        polygon_xy = list(zip(raw_points[0::2], raw_points[1::2]))
        self.ax.add_patch(
            Polygon(
                polygon_xy,
                closed=True,
                facecolor=fill_color(features),
                edgecolor="#222222",
                linewidth=2,
                linestyle="--" if is_dashed(features) else "-",
                zorder=1,
            )
        )

        arrow = movement_arrow(features, cx, cy, size)
        if arrow is not None:
            start_x, start_y, end_x, end_y = arrow
            self.ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 2},
                zorder=2,
            )
