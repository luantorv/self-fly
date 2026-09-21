from __future__ import annotations

import numpy as np

from self_fly.config.schema import LearnerConfig
from self_fly.stimuli.features import FEATURE_ORDER

from .actions import ACTIONS, Action


class LinearSoftmaxPolicy:
    """Softmax(W x + b) over the three actions. Linear and inspectable by
    design: weights are per (action, feature) pair."""

    def __init__(self, config: LearnerConfig, n_features: int = len(FEATURE_ORDER)):
        self.n_features = n_features
        self.W = np.full((len(ACTIONS), n_features), config.initial_weight_scale, dtype=float)
        self.b = np.zeros(len(ACTIONS), dtype=float)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = self.W @ features + self.b
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def sample(self, probs: np.ndarray, rng: np.random.Generator) -> Action:
        idx = rng.choice(len(ACTIONS), p=probs)
        return ACTIONS[idx]
