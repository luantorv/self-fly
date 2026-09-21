from __future__ import annotations

from enum import Enum


class StageOutcome(str, Enum):
    """How a stage concluded -- never silently coerced into looking like a
    real convergence. STABLE means the stability criterion actually fired.
    UNSTABLE means the trial budget ran out after at least some partial
    stable streak was observed (the policy was moving toward stability but
    didn't get there). TIMEOUT means the budget ran out without the policy
    ever holding a stable streak at all."""

    STABLE = "stable"
    UNSTABLE = "unstable"
    TIMEOUT = "timeout"
