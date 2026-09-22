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

    def evaluate(self, inputs_batch, conditioning_state=None) -> np.ndarray:
        """Action distributions for a batch of inputs, shape (N, 3), with
        NO side effects: the agent's internal state must be identical
        before and after. Every metric goes through here rather than
        through `probabilities`, because for a recurrent agent a forward
        pass IS experience -- measuring must not teach.

        All inputs are evaluated from the same `conditioning_state` (the
        agent's current state when None), so the result does not depend
        on the order of the batch."""
        ...

    def recurrent_state(self):
        """Copy of the internal recurrent state, or None for stateless
        architectures."""
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

    def parameters(self) -> dict:
        """Cheap numpy-level copy of the trainable parameters ONLY -- no
        recurrent state, no learner state, no list conversion. This is the
        hot path: D_t needs the previous parameters on every single trial,
        and round-tripping `full_state` through JSON-ready lists there
        would dominate the runtime."""
        ...

    def set_parameters(self, params: dict) -> None:
        """Inverse of `parameters`. Leaves recurrent and learner state
        untouched, so a temporary swap cannot disturb the trajectory."""
        ...

    def full_state(self) -> dict:
        """Complete, JSON-serializable state: every trainable parameter,
        the recurrent state if any, and the learner's own state. Unlike
        `snapshot_payload` (which is for inspection), this must be
        sufficient to RECONSTRUCT the agent so a restored run continues
        the same trajectory -- it is what checkpoint/restore and
        branching from a saved policy depend on."""
        ...

    def load_state(self, state: dict) -> None:
        """Inverse of `full_state`."""
        ...
