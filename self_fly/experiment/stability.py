from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from self_fly.config.schema import StabilityConfig

from .types import Trial


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between two empirical action distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@dataclass
class StabilityEvent:
    trial_index: int
    windows_compared: int
    action_distribution: dict[str, float]
    mean_reward: float


@dataclass
class _WindowStats:
    action_counts: Counter = field(default_factory=Counter)
    reward_sum: float = 0.0
    n: int = 0
    category_action_counts: dict = field(default_factory=lambda: defaultdict(Counter))

    def add(self, trial: Trial) -> None:
        self.action_counts[trial.action] += 1
        self.reward_sum += trial.reward
        self.n += 1
        self.category_action_counts[trial.category_id][trial.action] += 1

    def action_distribution(self) -> dict[str, float]:
        total = sum(self.action_counts.values()) or 1
        return {action: count / total for action, count in self.action_counts.items()}

    def mean_reward(self) -> float:
        return self.reward_sum / self.n if self.n else 0.0

    def category_distribution(self, category: str) -> dict[str, float]:
        counts = self.category_action_counts.get(category, Counter())
        total = sum(counts.values()) or 1
        return {action: count / total for action, count in counts.items()}


class StabilityDetector:
    """Declares a policy stable when, over `consecutive_windows_required`
    consecutive non-overlapping windows of `window_size` trials:

      (a) the empirical action distribution barely moves window-to-window
          (total variation < tv_threshold);
      (b) expected reward barely moves (|delta mean| < reward_delta_threshold)
          and its recent coefficient of variation stays < cv_threshold;
      (c) the action distribution restricted to each stimulus "equivalence
          category" is likewise consistent window-to-window
          (1 - mean category TV > consistency_threshold).

    All thresholds are experimental hypotheses, not implementation
    constants: they determine when pi_0 is captured and what counts as a
    stable policy, and must be reported alongside results.
    """

    def __init__(self, config: StabilityConfig):
        self.config = config
        self._current = _WindowStats()
        self._previous: _WindowStats | None = None
        self._recent_window_means: list[float] = []
        self._consecutive_stable = 0
        # Telemetry only -- read by StageMachine to tell "never got close to
        # stable" (TIMEOUT) apart from "was partway there and ran out of
        # trial budget" (UNSTABLE). Reset alongside everything else so it
        # reflects only the current stage's own history.
        self.max_consecutive_stable_observed = 0

    def update(self, trial: Trial) -> StabilityEvent | None:
        self._current.add(trial)
        if self._current.n < self.config.window_size:
            return None

        window = self._current
        self._current = _WindowStats()

        self._recent_window_means.append(window.mean_reward())
        max_len = self.config.consecutive_windows_required + 1
        self._recent_window_means = self._recent_window_means[-max_len:]

        if self._previous is None:
            self._previous = window
            return None

        stable_now = self._windows_are_stable(self._previous, window)
        self._previous = window
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
            )
        return None

    def reset(self) -> None:
        """Drop all accumulated window history. Called on a stage change so
        windows never straddle a stage boundary and stability is always
        measured against the new stage's own behaviour."""
        self._current = _WindowStats()
        self._previous = None
        self._recent_window_means = []
        self._consecutive_stable = 0
        self.max_consecutive_stable_observed = 0

    def _windows_are_stable(self, prev: _WindowStats, curr: _WindowStats) -> bool:
        tv = tv_distance(prev.action_distribution(), curr.action_distribution())
        if tv >= self.config.tv_threshold:
            return False

        reward_delta = abs(curr.mean_reward() - prev.mean_reward())
        if reward_delta >= self.config.reward_delta_threshold:
            return False

        if len(self._recent_window_means) >= 2:
            arr = np.array(self._recent_window_means)
            mean = arr.mean()
            cv = (arr.std() / abs(mean)) if mean != 0 else float("inf")
            if cv >= self.config.cv_threshold:
                return False

        categories = set(prev.category_action_counts) | set(curr.category_action_counts)
        if categories:
            tv_values = [
                tv_distance(prev.category_distribution(c), curr.category_distribution(c))
                for c in categories
            ]
            consistency_score = 1 - (sum(tv_values) / len(tv_values))
            if consistency_score <= self.config.consistency_threshold:
                return False

        return True
