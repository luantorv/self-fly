from __future__ import annotations

import tkinter as tk

from self_fly.stimuli.features import FeatureVector

from ..render import render_stimulus


class TkRenderer:
    """Wraps render.py's existing Tk drawing call in the StimulusRenderer
    shape -- identical behavior to calling render_stimulus directly."""

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas

    def render(self, features: FeatureVector, cx: float = 110, cy: float = 110, size: float = 70) -> None:
        render_stimulus(self.canvas, features, cx, cy, size)
