import numpy as np

from self_fly.config.schema import StimulusConfig
from self_fly.experiment.policy_distance import (
    build_reference_set,
    js_distance,
    policy_distance,
    tv_distance_arrays,
)
from self_fly.experiment.types import PolicySnapshot


def _snapshot(name: str, weights: np.ndarray, bias: np.ndarray) -> PolicySnapshot:
    return PolicySnapshot(
        name=name,
        stage=0,
        trial_index=0,
        weights=weights.tolist(),
        bias=bias.tolist(),
        reference_action_probs={},
        timestamp="t",
    )


def test_js_distance_zero_for_identical_distributions():
    p = np.array([0.7, 0.2, 0.1])
    assert js_distance(p, p) == 0.0


def test_js_distance_is_symmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.7])
    assert abs(js_distance(p, q) - js_distance(q, p)) < 1e-12


def test_js_distance_bounded_in_unit_interval():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    d = js_distance(p, q)
    assert 0.0 <= d <= 1.0


def test_tv_distance_zero_for_identical_distributions():
    p = np.array([0.5, 0.3, 0.2])
    assert tv_distance_arrays(p, p) == 0.0


def test_policy_distance_zero_for_identical_snapshots():
    rng = np.random.default_rng(0)
    reference_set = build_reference_set(StimulusConfig(reference_set_size=10), rng)
    weights = np.random.default_rng(1).normal(size=(3, 10))
    bias = np.zeros(3)
    snap = _snapshot("pi_x", weights, bias)

    result = policy_distance(snap, snap, reference_set)
    assert result["js"] == 0.0
    assert result["tv"] == 0.0


def test_policy_distance_large_for_very_different_weights():
    rng = np.random.default_rng(0)
    reference_set = build_reference_set(StimulusConfig(reference_set_size=20), rng)

    snap_a = _snapshot("pi_a", np.zeros((3, 10)), np.array([10.0, 0.0, 0.0]))
    snap_b = _snapshot("pi_b", np.zeros((3, 10)), np.array([0.0, 10.0, 0.0]))

    result = policy_distance(snap_a, snap_b, reference_set)
    assert result["js"] > 0.9
    assert result["tv"] > 0.9
