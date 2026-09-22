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

    def evaluate(self, inputs_batch, conditioning_state: np.ndarray | None = None) -> np.ndarray:
        """Action distributions for a batch of inputs. This agent is
        stateless, so `conditioning_state` is accepted (for interface
        parity with the recurrent agent) and ignored. Already free of
        side effects."""
        return np.array([self.probabilities(np.asarray(x, dtype=float)) for x in inputs_batch])

    def recurrent_state(self) -> np.ndarray | None:
        return None

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

    def parameters(self) -> dict:
        return {"W": self._state.policy.W.copy(), "b": self._state.policy.b.copy()}

    def set_parameters(self, params: dict) -> None:
        self._state.policy.W = params["W"].copy()
        self._state.policy.b = params["b"].copy()

    def full_state(self) -> dict:
        return {
            "agent_type": "baseline",
            "policy_W": self._state.policy.W.tolist(),
            "policy_b": self._state.policy.b.tolist(),
            "learner_baseline": self._state.learner.baseline.value,
            "learner_baseline_initialized": self._state.learner.baseline._initialized,
        }

    def load_state(self, state: dict) -> None:
        self._state.policy.W = np.array(state["policy_W"], dtype=float)
        self._state.policy.b = np.array(state["policy_b"], dtype=float)
        self._state.learner.baseline.value = state["learner_baseline"]
        self._state.learner.baseline._initialized = state["learner_baseline_initialized"]
