from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    incorrect: float = -1.0
    dont_know: float = -0.25


@dataclass(frozen=True)
class StabilityConfig:
    # Defaults calibrated empirically against this project's default
    # StimulusConfig/LearnerConfig: at window_size=100 a converged policy on
    # this (deliberately noisy, non-trivially-separable) task still shows
    # window-to-window TV ~0.06 (p90 ~0.13), reward-mean drift ~0.10 (p90
    # ~0.18) and per-category consistency as low as ~0.67 from sampling
    # noise alone. Thresholds are set with margin above/below those
    # sampling-noise figures so a converged policy reliably clears them
    # while a still-changing one reliably doesn't. Re-tune if
    # StimulusConfig.noise_std or window_size change materially.
    window_size: int = 100
    tv_threshold: float = 0.15
    reward_delta_threshold: float = 0.20
    cv_threshold: float = 0.35
    consistency_threshold: float = 0.60
    consecutive_windows_required: int = 3


@dataclass(frozen=True)
class StimulusConfig:
    signal_feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "shape_roundness": 1.0,
            "color_hue": -1.0,
            "texture_density": 0.8,
        }
    )
    noise_std: float = 0.35
    category_grid_size: int = 3
    reference_set_size: int = 50


@dataclass(frozen=True)
class LearnerConfig:
    learning_rate: float = 0.05
    baseline_momentum: float = 0.99
    initial_weight_scale: float = 0.0


@dataclass(frozen=True)
class StageConfig:
    stage: int
    stimulus_mix: dict[str, float]
    reward_overrides: dict[str, float] = field(default_factory=dict)
    entry_condition: str = "stability_detected"
    exit_condition: str = "stability_detected"
    max_trials: Optional[int] = None


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    run_name: str = "self_fly_run"
    reward: RewardConfig = field(default_factory=RewardConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    stages: list[StageConfig] = field(default_factory=list)
