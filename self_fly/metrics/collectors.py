from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from self_fly.experiment.entropy import action_entropy
from self_fly.experiment.types import Trial


@dataclass
class MetricsCollector:
    cumulative_reward: float = 0.0
    trial_count: int = 0
    reward_history: list = field(default_factory=list)
    action_counts: Counter = field(default_factory=Counter)
    action_counts_by_label: dict = field(default_factory=lambda: defaultdict(Counter))
    entropy_history: list = field(default_factory=list)

    def update(self, trial: Trial) -> None:
        self.trial_count += 1
        self.cumulative_reward += trial.reward
        self.reward_history.append(trial.reward)
        self.action_counts[trial.action] += 1
        self.action_counts_by_label[trial.ground_truth_label][trial.action] += 1
        self.entropy_history.append(action_entropy(trial.action_probs))

    def mean_entropy(self, last_n: int | None = None) -> float:
        history = self.entropy_history if last_n is None else self.entropy_history[-last_n:]
        return sum(history) / len(history) if history else 0.0

    def action_frequencies(self) -> dict[str, float]:
        total = sum(self.action_counts.values()) or 1
        return {action: count / total for action, count in self.action_counts.items()}

    def mean_reward(self, last_n: int | None = None) -> float:
        history = self.reward_history if last_n is None else self.reward_history[-last_n:]
        return sum(history) / len(history) if history else 0.0
