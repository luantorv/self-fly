from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Trial:
    trial_index: int
    stage: int
    stimulus_id: str
    category_id: str
    features: tuple[float, ...]
    action: str
    action_probs: tuple[float, float, float]
    reward: float
    ground_truth_label: str
    policy_weight_norm: float
    timestamp: str


@dataclass
class PolicySnapshot:
    name: str
    stage: int
    trial_index: int
    weights: list
    bias: list
    reference_action_probs: dict
    timestamp: str
