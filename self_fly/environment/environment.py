from __future__ import annotations

from .stimulus_types import GroundTruthStimulus, ObservableStimulus


class Environment:
    """The single isolation boundary between experiment truth and agent input."""

    @staticmethod
    def observe(ground_truth: GroundTruthStimulus) -> ObservableStimulus:
        return ObservableStimulus(features=ground_truth.features)
