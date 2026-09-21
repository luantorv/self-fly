from __future__ import annotations

import numpy as np

from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel

from .features import FEATURE_ORDER, FeatureVector

_SIGNED_FEATURES = ("orientation_sin", "orientation_cos", "movement_direction")


class SelfStimulusGenerator:
    """Derives a stimulus from a `signature` vector the agent itself
    computed from its own weights (see Agent.self_signature in
    agent/interface.py), re-encoded through the same [0,1]/[-1,1] ranges
    every other stimulus uses. The resulting FeatureVector is structurally
    indistinguishable from any other ObservableStimulus -- the agent's
    input pipeline has no way to tell it came from its own weights,
    because there is no SELF flag anywhere in it.

    This is agent-architecture-agnostic by design: what counts as "my own
    signature" is decided by each Agent implementation's self_signature()
    (BaselineAgent uses a weight-column norm; a different architecture
    could use something else entirely), never by this class. Whichever
    derivation is used is an EXPERIMENTAL HYPOTHESIS, not a neutral
    implementation choice, and must be reported alongside results.
    """

    def generate(
        self,
        signature: np.ndarray,
        stage: int,
        label: StimulusLabel = StimulusLabel.SELF_A,
    ) -> GroundTruthStimulus:
        max_value = signature.max() if signature.max() > 0 else 1.0
        normalized = signature / max_value

        values = []
        for name, value in zip(FEATURE_ORDER, normalized):
            if name in _SIGNED_FEATURES:
                values.append(float(value * 2.0 - 1.0))
            else:
                values.append(float(value))

        features = FeatureVector.from_tuple(tuple(values))
        return GroundTruthStimulus(
            stimulus_id=f"{label.value}_frozen",
            label=label,
            features=features,
            category_id=label.value,
            stage=stage,
        )
