from __future__ import annotations

import numpy as np

from self_fly.agent.policy import LinearSoftmaxPolicy
from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel

from .features import FEATURE_ORDER, FeatureVector

_SIGNED_FEATURES = ("orientation_sin", "orientation_cos", "movement_direction")


class SelfStimulusGenerator:
    """Derives a stimulus from the agent's own current policy weights: the
    per-feature column norm of W, re-encoded through the same [0,1]/[-1,1]
    ranges every other stimulus uses. The resulting FeatureVector is
    structurally indistinguishable from any other ObservableStimulus -- the
    agent's input pipeline has no way to tell it came from its own weights,
    because there is no SELF flag anywhere in it.

    This specific operationalisation (weight-column norms -> feature
    encoding) is an EXPERIMENTAL HYPOTHESIS, not a neutral implementation
    choice: a different derivation (e.g. a fixed identity signature, or a
    summary of past decisions) would change what "self-referential" means
    for this agent, and must be reported alongside results.
    """

    def generate(self, policy: LinearSoftmaxPolicy, stage: int) -> GroundTruthStimulus:
        norms = np.linalg.norm(policy.W, axis=0)
        max_norm = norms.max() if norms.max() > 0 else 1.0
        normalized = norms / max_norm

        values = []
        for name, value in zip(FEATURE_ORDER, normalized):
            if name in _SIGNED_FEATURES:
                values.append(float(value * 2.0 - 1.0))
            else:
                values.append(float(value))

        features = FeatureVector.from_tuple(tuple(values))
        return GroundTruthStimulus(
            stimulus_id="self_a_frozen",
            label=StimulusLabel.SELF_A,
            features=features,
            category_id="self_a",
            stage=stage,
        )
