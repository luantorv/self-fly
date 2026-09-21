from __future__ import annotations

import numpy as np

from self_fly.config.schema import LearnerConfig
from self_fly.stimuli.features import FEATURE_ORDER

from ..actions import Action
from .learner import REINFORCEUpdate
from .policy import LinearSoftmaxPolicy
from .state import AgentInternalState

_SIGNATURE_NORM_ORDER = {"l2_column_norm": 2, "l1_column_norm": 1}


class BaselineAgent:
    """Fase A's linear-softmax-policy + REINFORCE agent, wrapped to satisfy
    the Agent interface (interface.py). Behaviorally identical to the
    pre-Phase-B policy.py/learner.py/state.py combination -- this class
    only adds the self_signature/snapshot_payload surface engine.py needs
    to treat any agent architecture interchangeably."""

    def __init__(self, config: LearnerConfig, n_features: int = len(FEATURE_ORDER)):
        self._state = AgentInternalState(
            policy=LinearSoftmaxPolicy(config, n_features),
            learner=REINFORCEUpdate(config),
        )

    @property
    def policy(self) -> LinearSoftmaxPolicy:
        return self._state.policy

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return self._state.policy.probabilities(features)

    def sample(self, probs: np.ndarray, rng: np.random.Generator) -> Action:
        return self._state.policy.sample(probs, rng)

    def update(
        self, features: np.ndarray, action_idx: int, reward: float, probs: np.ndarray
    ) -> None:
        self._state.learner.update(self._state.policy, features, action_idx, reward, probs)

    def weight_norm_scalar(self) -> float:
        return self._state.weight_norm_scalar()

    def self_signature(self, transform: str = "l2_column_norm") -> np.ndarray:
        ord_ = _SIGNATURE_NORM_ORDER[transform]
        return np.linalg.norm(self._state.policy.W, ord=ord_, axis=0)

    def snapshot_payload(self) -> dict:
        return {
            "weights": self._state.policy.W.tolist(),
            "bias": self._state.policy.b.tolist(),
        }
