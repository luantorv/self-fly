import numpy as np

from self_fly.agent.baseline.learner import REINFORCEUpdate
from self_fly.agent.baseline.policy import LinearSoftmaxPolicy
from self_fly.config.schema import LearnerConfig


def test_policy_learns_simple_separable_task():
    rng = np.random.default_rng(0)
    config = LearnerConfig(learning_rate=0.1, baseline_momentum=0.9, initial_weight_scale=0.0)
    policy = LinearSoftmaxPolicy(config, n_features=10)
    learner = REINFORCEUpdate(config)

    def target_action_idx(features: np.ndarray) -> int:
        return 0 if features[0] > 0.5 else 1  # REAL if roundness>0.5 else FALSA

    def reward_for(action_idx: int, target_idx: int) -> float:
        return 1.0 if action_idx == target_idx else -1.0

    for _ in range(4000):
        features = rng.uniform(0.0, 1.0, size=10)
        probs = policy.probabilities(features)
        action_idx = rng.choice(3, p=probs)
        target_idx = target_action_idx(features)
        reward = reward_for(action_idx, target_idx)
        learner.update(policy, features, action_idx, reward, probs)

    n_eval = 500
    correct = 0
    for _ in range(n_eval):
        features = rng.uniform(0.0, 1.0, size=10)
        probs = policy.probabilities(features)
        target_idx = target_action_idx(features)
        correct += probs[target_idx] > 0.5
    accuracy = correct / n_eval
    assert accuracy > 0.85


def test_initial_policy_is_uniform():
    config = LearnerConfig(initial_weight_scale=0.0)
    policy = LinearSoftmaxPolicy(config, n_features=10)
    probs = policy.probabilities(np.zeros(10))
    assert np.allclose(probs, 1.0 / 3.0)


def test_sample_respects_rng_seed():
    config = LearnerConfig()
    policy = LinearSoftmaxPolicy(config, n_features=10)
    probs = np.array([0.7, 0.2, 0.1])
    a = policy.sample(probs, np.random.default_rng(42))
    b = policy.sample(probs, np.random.default_rng(42))
    assert a == b
