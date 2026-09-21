from __future__ import annotations

import math
from typing import Sequence

from .types import PolicySnapshot


def action_entropy(probs: Sequence[float]) -> float:
    """H(pi) = -sum(p * log2(p)) over the three actions. A zero-probability
    action contributes 0 (its log2(0) term is skipped), not NaN. Ranges
    from 0 (one action certain) to log2(3) (uniform over the three)."""
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


def mean_policy_entropy(snapshot: PolicySnapshot) -> float:
    """Mean action_entropy over every reference-set stimulus captured in
    this snapshot -- one scalar summary of how concentrated the policy is,
    computed purely from reference_action_probs so it works the same way
    regardless of which agent architecture produced the snapshot."""
    probs_list = list(snapshot.reference_action_probs.values())
    return sum(action_entropy(p) for p in probs_list) / len(probs_list)
