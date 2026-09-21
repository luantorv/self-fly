from __future__ import annotations

import numpy as np

from .actions import ACTIONS, Action


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> Action:
    """Shared by every agent implementation's own sample() -- action
    selection from a probability vector is the one piece of mechanics that
    doesn't vary with architecture."""
    idx = rng.choice(len(ACTIONS), p=probs)
    return ACTIONS[idx]
