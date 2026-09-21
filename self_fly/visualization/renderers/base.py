from __future__ import annotations

from typing import Protocol

from self_fly.stimuli.features import FeatureVector


class StimulusRenderer(Protocol):
    """Backend-agnostic contract every renderer (Tk, Matplotlib, a future
    3D one) satisfies. Only ever receives a FeatureVector -- same
    observer/agent boundary render.py already enforces, just made
    swappable: the engine and GUI don't need to know which backend is
    drawing."""

    def render(self, features: FeatureVector, **kwargs) -> None:
        ...
