from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from self_fly.config.schema import StabilityConfig

from .types import Trial


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between two empirical action distributions.
    No longer part of the stability criterion; kept because the reporting
    layer still uses it to describe behaviour."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@dataclass
class StabilityEvent:
    trial_index: int
    windows_compared: int
    action_distribution: dict[str, float]
    mean_reward: float
    accuracy: float
    median_policy_delta: float


@dataclass
class _WindowStats:
    action_counts: Counter = field(default_factory=Counter)
    reward_sum: float = 0.0
    n: int = 0
    category_action_counts: dict = field(default_factory=lambda: defaultdict(Counter))
    policy_deltas: list = field(default_factory=list)

    def add(self, trial: Trial, policy_delta: float | None) -> None:
        self.action_counts[trial.action] += 1
        self.reward_sum += trial.reward
        self.n += 1
        self.category_action_counts[trial.category_id][trial.action] += 1
        if policy_delta is not None:
            self.policy_deltas.append(policy_delta)

    def action_distribution(self) -> dict[str, float]:
        total = sum(self.action_counts.values()) or 1
        return {action: count / total for action, count in self.action_counts.items()}

    def mean_reward(self) -> float:
        return self.reward_sum / self.n if self.n else 0.0

    def median_policy_delta(self) -> float:
        return statistics.median(self.policy_deltas) if self.policy_deltas else float("inf")

    def category_distribution(self, category: str) -> dict[str, float]:
        counts = self.category_action_counts.get(category, Counter())
        total = sum(counts.values()) or 1
        return {action: count / total for action, count in counts.items()}


class StabilityDetector:
    """Declares a policy stable when, over `consecutive_windows_required`
    consecutive windows of `window_size` trials, BOTH hold:

      (A) task competence: accuracy on a fixed held-out REAL/FALSA
          validation set is at least `min_accuracy`;
      (B) policy settledness: the median of D_t = JS(pi_t, pi_{t-1}) over
          the window is at most `max_median_policy_delta`.

    Two things this deliberately does NOT do, both of them corrections:

    1. It never divides by the mean reward. The previous coefficient-of-
       variation criterion did, which made it ill-conditioned exactly
       where this experiment operates: adding zero-reward stimuli drags
       the mean toward zero and inflates the CV even when behaviour is
       unchanged, so the measuring instrument became stricter in
       proportion to the manipulation being studied.
    2. It never looks at the special stimulus categories. Stability is a
       statement about the REAL/FALSA task, so raising the proportion of
       special stimuli cannot by itself make stability harder to reach.

    Both thresholds are experimental hypotheses, fixed during a separate
    calibration phase and held constant across conditions afterwards.
    """

    def __init__(self, config: StabilityConfig, accuracy_fn=None):
        self.config = config
        # Called at most once per closed window; returns accuracy on the
        # fixed validation set. Injected so the detector never needs a
        # reference to the agent itself.
        self._accuracy_fn = accuracy_fn
        self._current = _WindowStats()
        self._previous_window_summary: tuple[float, float] | None = None
        self._consecutive_stable = 0
        # Telemetry only -- lets StageMachine tell "never got close to
        # stable" (TIMEOUT) apart from "was partway there" (UNSTABLE).
        self.max_consecutive_stable_observed = 0
        self.last_accuracy: float | None = None
        self.last_median_policy_delta: float | None = None

    def update(self, trial: Trial, policy_delta: float | None = None) -> StabilityEvent | None:
        self._current.add(trial, policy_delta)
        if self._current.n < self.config.window_size:
            return None

        window = self._current
        self._current = _WindowStats()

        accuracy = self._accuracy_fn() if self._accuracy_fn is not None else 1.0
        median_delta = window.median_policy_delta()
        self.last_accuracy = accuracy
        self.last_median_policy_delta = median_delta

        stable_now = (
            accuracy >= self.config.min_accuracy
            and median_delta <= self.config.max_median_policy_delta
        )
        self._consecutive_stable = self._consecutive_stable + 1 if stable_now else 0
        self.max_consecutive_stable_observed = max(
            self.max_consecutive_stable_observed, self._consecutive_stable
        )

        if self._consecutive_stable >= self.config.consecutive_windows_required:
            return StabilityEvent(
                trial_index=trial.trial_index,
                windows_compared=self._consecutive_stable,
                action_distribution=window.action_distribution(),
                mean_reward=window.mean_reward(),
                accuracy=accuracy,
                median_policy_delta=median_delta,
            )
        return None

    def reset(self) -> None:
        """Drop all accumulated window history. Called on a stage change so
        windows never straddle a stage boundary and stability is always
        measured against the new stage's own behaviour."""
        self._current = _WindowStats()
        self._previous_window_summary = None
        self._consecutive_stable = 0
        self.max_consecutive_stable_observed = 0
