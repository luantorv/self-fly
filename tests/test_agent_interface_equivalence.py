import numpy as np

from self_fly.agent.factory import make_agent
from self_fly.config.schema import ConnectomeConfig, LearnerConfig

_ACTION_VALUES = ("REAL", "FALSA", "NO_SE")


def _target_action_idx(features: np.ndarray) -> int:
    return 0 if features[0] > 0.5 else 1


def _reward_for(action_idx: int, target_idx: int) -> float:
    return 1.0 if action_idx == target_idx else -1.0


def _train_and_eval(agent, rng, n_train: int = 4000, n_eval: int = 500) -> float:
    for _ in range(n_train):
        features = rng.uniform(0.0, 1.0, size=10)
        probs = agent.probabilities(features)
        action = agent.sample(probs, rng)
        action_idx = _ACTION_VALUES.index(action.value)
        target_idx = _target_action_idx(features)
        agent.update(features, action_idx, _reward_for(action_idx, target_idx), probs)

    correct = 0
    for _ in range(n_eval):
        features = rng.uniform(0.0, 1.0, size=10)
        probs = agent.probabilities(features)
        correct += probs[_target_action_idx(features)] > 0.5
    return correct / n_eval


def test_baseline_agent_through_factory_learns_the_same_simple_task():
    """Certifies that moving policy.py/learner.py/state.py into
    agent/baseline/ and wrapping them in BaselineAgent changed nothing
    behavioral -- same task, same >0.85 bar as test_policy.py's original
    direct-construction version."""
    rng = np.random.default_rng(0)
    agent = make_agent(
        "baseline", LearnerConfig(learning_rate=0.1, baseline_momentum=0.9), ConnectomeConfig()
    )
    assert _train_and_eval(agent, rng) > 0.85


def test_baseline_agent_is_deterministic_given_the_same_seed():
    def run() -> np.ndarray:
        rng = np.random.default_rng(7)
        agent = make_agent("baseline", LearnerConfig(), ConnectomeConfig())
        for _ in range(200):
            features = rng.uniform(0.0, 1.0, size=10)
            probs = agent.probabilities(features)
            action = agent.sample(probs, rng)
            action_idx = _ACTION_VALUES.index(action.value)
            agent.update(features, action_idx, 1.0, probs)
        return agent.policy.W.copy()

    assert np.allclose(run(), run())


def test_connectome_agent_exposes_the_same_interface_without_exceptions():
    rng = np.random.default_rng(1)
    agent = make_agent("connectome", LearnerConfig(), ConnectomeConfig(), seed=42)
    for _ in range(200):
        features = rng.uniform(0.0, 1.0, size=10)
        probs = agent.probabilities(features)
        action = agent.sample(probs, rng)
        action_idx = _ACTION_VALUES.index(action.value)
        agent.update(features, action_idx, 1.0, probs)

    assert agent.weight_norm_scalar() >= 0.0
    assert agent.self_signature().shape == (10,)
    payload = agent.snapshot_payload()
    assert "weights" in payload and "bias" in payload
