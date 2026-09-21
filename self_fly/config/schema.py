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
class ConnectomeConfig:
    # hidden_dim/recurrent_leak/initial_weight_scale only matter for
    # ConnectomeInspiredAgent; learning_rate/baseline_momentum mirror
    # LearnerConfig's fields because its ActionValuation layer reuses
    # agent/baseline/learner.py's REINFORCEUpdate directly (see that
    # class's docstring for why).
    hidden_dim: int = 8
    recurrent_leak: float = 0.1
    initial_weight_scale: float = 0.05
    learning_rate: float = 0.05
    baseline_momentum: float = 0.99


@dataclass(frozen=True)
class SelfVariantConfig:
    # "l2_column_norm" is SELF_A's original transform (SelfStimulusGenerator);
    # "l1_column_norm" is SELF_B's -- a deliberately slightly different,
    # equally arbitrary aggregation of the same weight matrix, not "a
    # better" one. other_agent_seed_offset keeps the auxiliary OTHER
    # agent's RNG stream fully disjoint from the main experiment's.
    self_a_transform: str = "l2_column_norm"
    self_b_transform: str = "l1_column_norm"
    other_agent_seed_offset: int = 10_000


@dataclass(frozen=True)
class ControlStimulusConfig:
    # Placeholder for now (block 6 needs only "frozen once, in-distribution
    # features" -- see ControlStimulusGenerator). Left as its own dataclass,
    # not inlined, so later variants (e.g. re-sampled per trial) have
    # somewhere to add fields without touching ExperimentConfig again.
    freeze_at_pi0: bool = True


@dataclass(frozen=True)
class ConflictConfig:
    """Reward for maintaining vs. changing the last non-NO_SE action taken
    toward each of `categories` (tracked per category_id, not globally --
    see RewardModel._last_action_by_category). Every field is an
    experimental hypothesis about how sharply to reward/punish
    consistency, not a "correct" value -- report alongside results."""

    maintain_reward: float
    change_reward: float
    no_se_reward: float
    first_exposure_reward: float = 0.0
    categories: tuple[str, ...] = ("self_a",)


@dataclass(frozen=True)
class StageConfig:
    stage: int
    stimulus_mix: dict[str, float]
    reward_overrides: dict[str, float] = field(default_factory=dict)
    entry_condition: str = "stability_detected"
    exit_condition: str = "stability_detected"
    max_trials: Optional[int] = None
    conflict: Optional[ConflictConfig] = None


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    run_name: str = "self_fly_run"
    reward: RewardConfig = field(default_factory=RewardConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    stages: list[StageConfig] = field(default_factory=list)
    # Phase B: which agent implementation and which experimental condition
    # produced this run. Both are plain strings (not enums) so a config.json
    # from before these fields existed still loads via config_from_dict's
    # .get(..., "baseline") fallback.
    agent_type: str = "baseline"
    experimental_condition: str = "baseline"
    control_stimulus: ControlStimulusConfig = field(default_factory=ControlStimulusConfig)
    self_variant: SelfVariantConfig = field(default_factory=SelfVariantConfig)
    connectome: ConnectomeConfig = field(default_factory=ConnectomeConfig)
