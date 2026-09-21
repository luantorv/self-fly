import numpy as np

from self_fly.config.schema import StimulusConfig
from self_fly.environment.stimulus_types import StimulusLabel
from self_fly.stimuli.features import FEATURE_ORDER
from self_fly.stimuli.generator import StimulusGenerator


def _best_single_feature_accuracy(values: np.ndarray, is_real: np.ndarray) -> float:
    thresholds = np.linspace(values.min(), values.max(), 50)
    best = 0.0
    for t in thresholds:
        pred_a = values > t
        acc_a = np.mean(pred_a == is_real)
        best = max(best, acc_a, 1.0 - acc_a)
    return best


def test_no_single_feature_is_a_trivial_shortcut():
    rng = np.random.default_rng(123)
    gen = StimulusGenerator(StimulusConfig(), rng)
    stimuli = [gen.sample(stage=0) for _ in range(4000)]
    is_real = np.array([s.label == StimulusLabel.REAL for s in stimuli])

    assert 0.25 < is_real.mean() < 0.75

    for i, name in enumerate(FEATURE_ORDER):
        values = np.array([s.features.to_tuple()[i] for s in stimuli])
        acc = _best_single_feature_accuracy(values, is_real)
        assert acc < 0.85, f"feature {name} alone predicts label with accuracy {acc:.2f}"


def test_category_ids_group_multiple_stimuli():
    rng = np.random.default_rng(7)
    gen = StimulusGenerator(StimulusConfig(), rng)
    stimuli = [gen.sample(stage=0) for _ in range(500)]
    categories = {s.category_id for s in stimuli}
    assert 1 < len(categories) < 500


def test_stimulus_ids_are_unique():
    rng = np.random.default_rng(1)
    gen = StimulusGenerator(StimulusConfig(), rng)
    ids = [gen.sample(stage=0).stimulus_id for _ in range(200)]
    assert len(set(ids)) == 200
