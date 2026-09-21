from __future__ import annotations

from typing import Protocol

import numpy as np

from .actions import Action


class Agent(Protocol):
    """Everything ExperimentEngine needs from an agent implementation --
    baseline (linear softmax + REINFORCE) or connectome-inspired,
    interchangeably, via make_agent() in factory.py. Only ever receives
    `features`: nothing in this interface can carry a privileged label,
    category, or the experiment's stage number (enforced across every
    file under agent/ by test_no_privileged_leak.py's AST check)."""

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        ...

    def sample(self, probs: np.ndarray, rng: np.random.Generator) -> Action:
        ...

    def update(
        self, features: np.ndarray, action_idx: int, reward: float, probs: np.ndarray
    ) -> None:
        ...

    def weight_norm_scalar(self) -> float:
        ...

    def self_signature(self, transform: str = "l2_column_norm") -> np.ndarray:
        """Length len(FEATURE_ORDER) vector this agent's own SELF stimulus
        is derived from. What "self-referential" means is architecture-
        specific and, either way, an experimental hypothesis rather than a
        neutral choice -- see stimuli/self_stimulus.py."""
        ...

    def snapshot_payload(self) -> dict:
        """Serializable dict for PolicySnapshot.weights/bias -- whatever
        this architecture wants persisted for later inspection. Snapshot
        comparisons across agent types must use
        policy_distance.policy_distance_from_snapshots (reference_action_probs
        based), never policy_distance.policy_distance (which assumes a
        linear softmax and only ever gets BaselineAgent snapshots)."""
        ...
