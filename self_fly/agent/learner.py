from __future__ import annotations

import numpy as np

from self_fly.config.schema import LearnerConfig

from .policy import LinearSoftmaxPolicy


class RunningMeanBaseline:
    """Exponential moving average of reward, used to reduce update variance."""

    def __init__(self, momentum: float):
        self.momentum = momentum
        self.value = 0.0
        self._initialized = False

    def update(self, reward: float) -> None:
        if not self._initialized:
            self.value = reward
            self._initialized = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * reward


class REINFORCEUpdate:
    """One-step policy-gradient update: advantage-weighted softmax gradient."""

    def __init__(self, config: LearnerConfig):
        self.learning_rate = config.learning_rate
        self.baseline = RunningMeanBaseline(config.baseline_momentum)

    def update(
        self,
        policy: LinearSoftmaxPolicy,
        features: np.ndarray,
        action_idx: int,
        reward: float,
        probs: np.ndarray,
    ) -> None:
        advantage = reward - self.baseline.value
        grad = -probs.copy()
        grad[action_idx] += 1.0
        policy.W += self.learning_rate * advantage * np.outer(grad, features)
        policy.b += self.learning_rate * advantage * grad
        self.baseline.update(reward)
