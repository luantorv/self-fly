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


def render_stimulus(canvas: tk.Canvas, features: FeatureVector, cx: float, cy: float, size: float) -> None:
    """Draws an abstract geometric stimulus from a FeatureVector. This is
    the ONLY rendering path, used for every stimulus including the
    self-referential one -- there is no separate visual branch for it."""
    canvas.delete("stimulus")

    bg_shade = int(235 - features.context_level * 70)
    bg_color = f"#{bg_shade:02x}{bg_shade:02x}{bg_shade:02x}"
    canvas.create_rectangle(
        cx - size * 1.3, cy - size * 1.3, cx + size * 1.3, cy + size * 1.3,
        fill=bg_color, outline="", tags="stimulus",
    )

    n_sides = 3 + round(features.shape_roundness * 9)
    angle_offset = math.atan2(features.orientation_sin, features.orientation_cos)
    width_scale = 0.5 + features.aspect_ratio * 0.5
    height_scale = 1.0 - features.aspect_ratio * 0.3

    points: list[float] = []
    for i in range(n_sides):
        theta = angle_offset + 2 * math.pi * i / n_sides
        points.append(cx + size * width_scale * math.cos(theta))
        points.append(cy + size * height_scale * math.sin(theta))

    fill_color = _hsv_to_hex(features.color_hue, 0.3 + 0.6 * features.color_saturation)
    dash = None if features.texture_density < 0.5 else (4, 3)
    canvas.create_polygon(
        points, fill=fill_color, outline="#222222", width=2, dash=dash, tags="stimulus",
    )

    if features.movement_speed > 0.05:
        arrow_len = size * (0.4 + features.movement_speed * 0.8)
        direction_angle = features.movement_direction * math.pi
        end_x = cx + arrow_len * math.cos(direction_angle)
        end_y = cy + arrow_len * math.sin(direction_angle)
        canvas.create_line(
            cx, cy, end_x, end_y, arrow=tk.LAST, width=2, fill="#444444", tags="stimulus",
        )
