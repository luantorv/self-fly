from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .learner import REINFORCEUpdate
from .policy import LinearSoftmaxPolicy


@dataclass
class AgentInternalState:
    """Everything the agent actually maintains: policy weights and the
    learner's baseline. Nothing here is derived from experiment ground truth."""

    policy: LinearSoftmaxPolicy
    learner: REINFORCEUpdate

    def weight_norm_per_feature(self) -> np.ndarray:
        return np.linalg.norm(self.policy.W, axis=0)

    def weight_norm_scalar(self) -> float:
        return float(np.linalg.norm(self.policy.W))
