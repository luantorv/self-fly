from __future__ import annotations

from self_fly.stimuli.features import FeatureVector


class Renderer3D:
    """Placeholder for a future 3D stimulus renderer. Deliberately not
    implemented yet -- this exists only to declare the extension point
    (StimulusRenderer) now, without committing to a 3D implementation
    before it's actually needed."""

    def render(self, features: FeatureVector, **kwargs) -> None:
        raise NotImplementedError("3D rendering is a later Phase B stage, not implemented yet")
