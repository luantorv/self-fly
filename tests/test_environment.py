import dataclasses

import numpy as np

from self_fly.config.schema import StimulusConfig
from self_fly.environment.environment import Environment
from self_fly.environment.stimulus_types import ObservableStimulus
from self_fly.stimuli.generator import StimulusGenerator


def test_observable_stimulus_has_no_label_field():
    field_names = {f.name for f in dataclasses.fields(ObservableStimulus)}
    assert field_names == {"features"}


def test_observe_strips_everything_but_features():
    rng = np.random.default_rng(0)
    gen = StimulusGenerator(StimulusConfig(), rng)
    ground_truth = gen.sample(stage=0)

    observable = Environment.observe(ground_truth)

    assert observable.features == ground_truth.features
    assert not hasattr(observable, "label")
    assert not hasattr(observable, "category_id")
    assert not hasattr(observable, "stage")
