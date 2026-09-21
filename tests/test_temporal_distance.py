import numpy as np

from self_fly.analysis.temporal_dynamics import classify_dt_series
from self_fly.config.defaults import default_config
from self_fly.experiment.engine import ExperimentEngine
from self_fly.experiment.policy_distance import TemporalDistanceTracker


def test_first_step_returns_none():
    tracker = TemporalDistanceTracker(reference_set=[np.zeros(10)])
    assert tracker.step(lambda f: np.array([1 / 3, 1 / 3, 1 / 3])) is None


def test_step_returns_zero_when_policy_unchanged():
    reference_set = [np.zeros(10), np.ones(10)]
    tracker = TemporalDistanceTracker(reference_set=reference_set)
    fixed_probs = lambda f: np.array([0.5, 0.3, 0.2])

    tracker.step(fixed_probs)
    delta = tracker.step(fixed_probs)

    assert delta == 0.0


def test_step_returns_positive_after_a_real_change():
    tracker = TemporalDistanceTracker(reference_set=[np.zeros(10)])
    tracker.step(lambda f: np.array([1.0, 0.0, 0.0]))
    delta = tracker.step(lambda f: np.array([0.0, 1.0, 0.0]))

    assert delta > 0.0


def test_engine_trial_has_policy_js_delta_from_second_trial_onward():
    engine = ExperimentEngine(default_config(seed=0))
    trial0 = engine.step()
    assert trial0.policy_js_delta is None

    trial1 = engine.step()
    assert trial1.policy_js_delta is not None
    assert trial1.policy_js_delta >= 0.0


def test_classify_dt_series_handles_empty_input():
    assert classify_dt_series([]) == {"n": 0}


def test_classify_dt_series_reports_expected_keys_and_flags_a_spike():
    series = [0.05] * 40 + [0.95] + [0.05] * 40
    result = classify_dt_series(series, window=10)

    assert result["n"] == 81
    assert result["abrupt_change_count"] >= 1
    assert "window_means" in result
    assert "drift_slope" in result
    assert "tail_cv" in result
