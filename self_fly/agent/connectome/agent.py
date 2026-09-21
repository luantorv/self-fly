from __future__ import annotations

import numpy as np

from self_fly.config.schema import ConnectomeConfig, LearnerConfig
from self_fly.stimuli.features import FEATURE_ORDER

from ..actions import ACTIONS, Action
from ..baseline.learner import REINFORCEUpdate
from ..common import sample_action
from .layers import ActionValuation, RecurrentIntermediate, SensoryProcessing

_SIGNATURE_NORM_ORDER = {"l2_column_norm": 2, "l1_column_norm": 1}


class ConnectomeInspiredAgent:
    """Reduced connectome-inspired architecture:

        input -> SensoryProcessing -> RecurrentIntermediate (genuine
        recurrent memory across trials) -> ActionValuation ->
        ActionSelection (softmax + sample) -> RewardModulation
        (REINFORCE, baseline-subtracted).

    Each layer's biological motivation is documented on the layer itself
    (layers.py); none of them claim fidelity to any specific circuit --
    "inspired by" means the pipeline shape, not a reproduction.

    Design simplification, stated explicitly rather than smuggled in:
    SensoryProcessing and RecurrentIntermediate are randomly initialized
    and held FIXED (reservoir-computing style). Only ActionValuation is
    trained, via agent/baseline/learner.py's REINFORCEUpdate reused
    directly (not reimplemented), applied to the hidden_state the fixed
    reservoir produces instead of raw features. This keeps training as
    simple as BaselineAgent's while still giving the agent genuine
    trial-to-trial memory through the recurrent hidden state.
    """

    def __init__(
        self,
        config: ConnectomeConfig,
        n_features: int = len(FEATURE_ORDER),
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        n_actions = len(ACTIONS)
        self.sensory = SensoryProcessing(n_features, config.hidden_dim, rng, config.initial_weight_scale)
        self.recurrent = RecurrentIntermediate(
            config.hidden_dim, config.recurrent_leak, rng, config.initial_weight_scale
        )
        self.valuation = ActionValuation(config.hidden_dim, n_actions, rng, config.initial_weight_scale)
        self._learner = REINFORCEUpdate(
            LearnerConfig(learning_rate=config.learning_rate, baseline_momentum=config.baseline_momentum)
        )
        self._last_hidden_state = np.zeros(config.hidden_dim)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        sensory_output = self.sensory.forward(features)
        hidden_state = self.recurrent.forward(sensory_output)
        self._last_hidden_state = hidden_state
        return self.valuation.probabilities(hidden_state)

    def sample(self, probs: np.ndarray, rng: np.random.Generator) -> Action:
        return sample_action(probs, rng)

    def update(
        self, features: np.ndarray, action_idx: int, reward: float, probs: np.ndarray
    ) -> None:
        # Reuses the hidden_state already advanced by probabilities() for
        # this same trial -- recomputing via a fresh forward pass here
        # would mutate the recurrent state a second time (forward() is a
        # leaky-integrator step, not idempotent).
        self._learner.update(self.valuation, self._last_hidden_state, action_idx, reward, probs)

    def weight_norm_scalar(self) -> float:
        return float(
            np.linalg.norm(self.valuation.W)
            + np.linalg.norm(self.sensory.W)
            + np.linalg.norm(self.recurrent.W_rec)
            + np.linalg.norm(self.recurrent.W_in)
        )

    def self_signature(self, transform: str = "l2_column_norm") -> np.ndarray:
        ord_ = _SIGNATURE_NORM_ORDER[transform]
        return np.linalg.norm(self.sensory.W, ord=ord_, axis=0)

    def snapshot_payload(self) -> dict:
        # Same "weights"/"bias" shape as BaselineAgent -- PolicySnapshot's
        # schema stays agent-agnostic. Only the trained layer (ActionValuation)
        # is exposed; SensoryProcessing/RecurrentIntermediate are fixed for
        # the whole run and not meaningful to compare snapshot-to-snapshot.
        return {
            "weights": self.valuation.W.tolist(),
            "bias": self.valuation.b.tolist(),
        }
