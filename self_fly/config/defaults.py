from __future__ import annotations

from .schema import (
    ExperimentConfig,
    LearnerConfig,
    RewardConfig,
    StabilityConfig,
    StageConfig,
    StimulusConfig,
)


def default_config(seed: int = 0, run_name: str = "self_fly_run") -> ExperimentConfig:
    stage0 = StageConfig(
        stage=0,
        stimulus_mix={"real": 0.5, "falsa": 0.5},
        reward_overrides={},
        entry_condition="initial",
        exit_condition="stability_detected",
    )
    stage1 = StageConfig(
        stage=1,
        stimulus_mix={"real": 0.4, "falsa": 0.4, "self_a": 0.2},
        reward_overrides={
            "self_a|REAL": 0.0,
            "self_a|FALSA": 0.0,
            "self_a|NO_SE": 0.0,
        },
        entry_condition="stability_detected",
        exit_condition="stability_detected_or_max_trials",
        max_trials=5000,
    )
    return ExperimentConfig(
        seed=seed,
        run_name=run_name,
        reward=RewardConfig(),
        stability=StabilityConfig(),
        stimulus=StimulusConfig(),
        learner=LearnerConfig(),
        stages=[stage0, stage1],
    )
