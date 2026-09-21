from __future__ import annotations

import colorsys
import math
import tkinter as tk

from self_fly.stimuli.features import FeatureVector


def _hsv_to_hex(hue: float, saturation: float, value: float = 0.85) -> str:
    hue = hue % 1.0
    saturation = min(max(saturation, 0.0), 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# Pure geometry/color functions, backend-agnostic (no tkinter/matplotlib
# import here) -- kept in this file rather than a new module so the
# privileged-info AST check (test_no_privileged_leak.py) only ever needs
# to cover this one file for the "how a stimulus is drawn" logic, not a
# growing list of renderer modules. self_fly/visualization/renderers/*.py
# import these to build backend-specific renderers on top.


def background_color(features: FeatureVector) -> str:
    bg_shade = int(235 - features.context_level * 70)
    return f"#{bg_shade:02x}{bg_shade:02x}{bg_shade:02x}"


def polygon_points(features: FeatureVector, cx: float, cy: float, size: float) -> list[float]:
    n_sides = 3 + round(features.shape_roundness * 9)
    angle_offset = math.atan2(features.orientation_sin, features.orientation_cos)
    width_scale = 0.5 + features.aspect_ratio * 0.5
    height_scale = 1.0 - features.aspect_ratio * 0.3

    points: list[float] = []
    for i in range(n_sides):
        theta = angle_offset + 2 * math.pi * i / n_sides
        points.append(cx + size * width_scale * math.cos(theta))
        points.append(cy + size * height_scale * math.sin(theta))
    return points


def fill_color(features: FeatureVector) -> str:
    return _hsv_to_hex(features.color_hue, 0.3 + 0.6 * features.color_saturation)


def is_dashed(features: FeatureVector) -> bool:
    return features.texture_density >= 0.5


def movement_arrow(
    features: FeatureVector, cx: float, cy: float, size: float
) -> tuple[float, float, float, float] | None:
    """(start_x, start_y, end_x, end_y), or None if movement_speed is too
    small to bother drawing."""
    if features.movement_speed <= 0.05:
        return None
    arrow_len = size * (0.4 + features.movement_speed * 0.8)
    direction_angle = features.movement_direction * math.pi
    end_x = cx + arrow_len * math.cos(direction_angle)
    end_y = cy + arrow_len * math.sin(direction_angle)
    return (cx, cy, end_x, end_y)


def render_stimulus(canvas: tk.Canvas, features: FeatureVector, cx: float, cy: float, size: float) -> None:
    """Draws an abstract geometric stimulus from a FeatureVector onto a
    tk.Canvas. This is the ONLY rendering path used for every stimulus
    including the self-referential one -- there is no separate visual
    branch for it. A thin wrapper over this module's pure functions above;
    see visualization/renderers/tk_renderer.py for the StimulusRenderer-
    shaped equivalent."""
    canvas.delete("stimulus")

    canvas.create_rectangle(
        cx - size * 1.3, cy - size * 1.3, cx + size * 1.3, cy + size * 1.3,
        fill=background_color(features), outline="", tags="stimulus",
    )

    canvas.create_polygon(
        polygon_points(features, cx, cy, size),
        fill=fill_color(features),
        outline="#222222",
        width=2,
        dash=(4, 3) if is_dashed(features) else None,
        tags="stimulus",
    )

    arrow = movement_arrow(features, cx, cy, size)
    if arrow is not None:
        start_x, start_y, end_x, end_y = arrow
        canvas.create_line(
            start_x, start_y, end_x, end_y, arrow=tk.LAST, width=2, fill="#444444", tags="stimulus",
        )
