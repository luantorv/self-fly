from __future__ import annotations

import numpy as np

from self_fly.config.schema import StimulusConfig
from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel

from .generator import StimulusGenerator


class ControlStimulusGenerator:
    """Frozen stimulus with i.i.d., in-distribution features -- sampled the
    same way (and, when used, at the same moment) as SELF_A is frozen, but
    with its own dedicated RNG stream so enabling the control condition
    never perturbs the main experiment's trial sequence.

    This isolates one confound from another: SELF_A is both (a) frozen
    with a fixed zero reward and (b) structurally out-of-distribution
    (self_stimulus.py's own docstring flags this). CONTROL shares (a) but
    not (b), so a difference in post-freeze dynamics between the self and
    control conditions can be attributed to being out-of-distribution
    specifically, not merely to being frozen. EXPERIMENTAL HYPOTHESIS, not
    a neutral choice -- report alongside results.
    """

    def __init__(self, stimulus_config: StimulusConfig, rng: np.random.Generator):
        self._generator = StimulusGenerator(stimulus_config, rng)

    def generate(self, stage: int) -> GroundTruthStimulus:
        sample = self._generator.sample(stage)
        return GroundTruthStimulus(
            stimulus_id="control_frozen",
            label=StimulusLabel.CONTROL,
            features=sample.features,
            category_id="control",
            stage=stage,
        )
