import numpy as np

from self_fly.config.schema import StimulusConfig
from self_fly.environment.environment import Environment
from self_fly.environment.stimulus_types import StimulusLabel
from self_fly.stimuli.control_stimulus import ControlStimulusGenerator


def test_control_stimulus_has_control_label_and_category():
    generator = ControlStimulusGenerator(StimulusConfig(), np.random.default_rng(0))
    stimulus = generator.generate(stage=1)
    assert stimulus.label == StimulusLabel.CONTROL
    assert stimulus.category_id == "control"
    assert stimulus.stage == 1


def test_control_stimulus_features_are_in_distribution_not_near_a_vertex():
    """Unlike SELF_A (near a vertex of the hypercube by construction -- one
    feature pinned near 1.0, the rest near 0/-1), the control stimulus is a
    plain draw from the same i.i.d. distribution as REAL/FALSA stimuli."""
    generator = ControlStimulusGenerator(StimulusConfig(), np.random.default_rng(1))
    stimulus = generator.generate(stage=1)
    values = stimulus.features.to_tuple()
    near_extreme = [abs(v) < 0.05 or abs(abs(v) - 1.0) < 0.05 for v in values]
    assert not all(near_extreme)


def test_control_stimulus_never_carries_label_through_observe():
    generator = ControlStimulusGenerator(StimulusConfig(), np.random.default_rng(2))
    stimulus = generator.generate(stage=1)
    observable = Environment.observe(stimulus)
    assert not hasattr(observable, "label")
    assert not hasattr(observable, "category_id")
    assert not hasattr(observable, "stage")
