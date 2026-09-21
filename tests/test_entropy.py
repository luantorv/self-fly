import math

from self_fly.experiment.entropy import action_entropy, mean_policy_entropy
from self_fly.experiment.types import PolicySnapshot


def test_uniform_distribution_has_max_entropy():
    assert math.isclose(action_entropy([1 / 3, 1 / 3, 1 / 3]), math.log2(3))


def test_degenerate_distribution_has_zero_entropy():
    assert action_entropy([1.0, 0.0, 0.0]) == 0.0


def test_entropy_is_symmetric_under_permutation():
    a = action_entropy([0.7, 0.2, 0.1])
    b = action_entropy([0.1, 0.7, 0.2])
    assert math.isclose(a, b)


def test_entropy_bounded_between_zero_and_log2_3():
    for probs in ([0.5, 0.3, 0.2], [0.9, 0.05, 0.05], [1.0, 0.0, 0.0]):
        h = action_entropy(probs)
        assert 0.0 <= h <= math.log2(3) + 1e-9


def _snapshot(probs_by_ref: dict) -> PolicySnapshot:
    return PolicySnapshot(
        name="pi_test",
        stage=0,
        trial_index=0,
        weights=[[0.0] * 10] * 3,
        bias=[0.0, 0.0, 0.0],
        reference_action_probs=probs_by_ref,
        timestamp="t",
    )


def test_mean_policy_entropy_averages_over_reference_set():
    snapshot = _snapshot({"ref0": (1.0, 0.0, 0.0), "ref1": (1 / 3, 1 / 3, 1 / 3)})
    expected = (action_entropy((1.0, 0.0, 0.0)) + action_entropy((1 / 3, 1 / 3, 1 / 3))) / 2
    assert math.isclose(mean_policy_entropy(snapshot), expected)
