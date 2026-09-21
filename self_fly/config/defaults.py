from __future__ import annotations

from dataclasses import replace

from .schema import (
    ConflictConfig,
    ExperimentConfig,
    LearnerConfig,
    RewardConfig,
    SelfVariantConfig,
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


def condition_control_config(
    seed: int = 0, run_name: str = "self_fly_control"
) -> ExperimentConfig:
    """Same shape as default_config's stage 1 (same self/control frequency
    -- 20% -- and the same fixed zero reward), but the 20% slot is filled
    by a frozen, in-distribution CONTROL stimulus instead of the
    structurally out-of-distribution SELF_A. See ControlStimulusGenerator
    for why this isolates one confound from the other."""
    stage0 = StageConfig(
        stage=0,
        stimulus_mix={"real": 0.5, "falsa": 0.5},
        reward_overrides={},
        entry_condition="initial",
        exit_condition="stability_detected",
    )
    stage1 = StageConfig(
        stage=1,
        stimulus_mix={"real": 0.4, "falsa": 0.4, "control": 0.2},
        reward_overrides={
            "control|REAL": 0.0,
            "control|FALSA": 0.0,
            "control|NO_SE": 0.0,
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
        agent_type="baseline",
        experimental_condition="control",
    )


def condition_self_config(seed: int = 0, run_name: str = "self_fly_self") -> ExperimentConfig:
    """default_config(), relabeled explicitly as the "self" condition (C)
    for block 9's A/B/C/D comparisons -- the underlying stage mechanics
    are identical to default_config(); only experimental_condition differs."""
    return replace(
        default_config(seed=seed, run_name=run_name), experimental_condition="self"
    )


def condition_multi_self_config(
    seed: int = 0, run_name: str = "self_fly_self_multi"
) -> ExperimentConfig:
    """SELF_A, SELF_B (a slightly different weight-signature transform),
    and OTHER (a separately-trained auxiliary agent's signature) share the
    40% of stage 1 that default_config() gives to SELF_A alone. None of
    the three ever reaches agent/ with a label -- see engine.py's
    _sample_stimulus / environment.observe()."""
    stage0 = StageConfig(
        stage=0,
        stimulus_mix={"real": 0.5, "falsa": 0.5},
        reward_overrides={},
        entry_condition="initial",
        exit_condition="stability_detected",
    )
    third = 0.4 / 3
    stage1 = StageConfig(
        stage=1,
        stimulus_mix={"real": 0.3, "falsa": 0.3, "self_a": third, "self_b": third, "other": third},
        reward_overrides={
            "self_a|REAL": 0.0,
            "self_a|FALSA": 0.0,
            "self_a|NO_SE": 0.0,
            "self_b|REAL": 0.0,
            "self_b|FALSA": 0.0,
            "self_b|NO_SE": 0.0,
            "other|REAL": 0.0,
            "other|FALSA": 0.0,
            "other|NO_SE": 0.0,
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
        agent_type="baseline",
        experimental_condition="self_multi",
        self_variant=SelfVariantConfig(),
    )


def phase_b_conflict_stages(base_stage1: StageConfig) -> list[StageConfig]:
    """Stage 2 (mild): maintaining the last action toward SELF_A is
    slightly rewarded, changing it has a small cost. Stage 3 (strong): the
    two rewards flip sign -- maintaining turns negative, changing turns
    positive. Stage 4 (three-way): maintaining is strongly penalized,
    changing is moderately rewarded, NO_SE carries a small penalty, and
    the conflict now spans SELF_A/SELF_B/OTHER at once instead of SELF_A
    alone. Every reward value here is an arbitrary experimental hypothesis
    about how sharp the conflict should be, calibrated for nothing --
    report alongside results, do not treat as validated."""
    stage2 = StageConfig(
        stage=2,
        stimulus_mix=dict(base_stage1.stimulus_mix),
        entry_condition="stability_detected",
        exit_condition="stability_detected_or_max_trials",
        max_trials=5000,
        conflict=ConflictConfig(
            maintain_reward=0.1,
            change_reward=-0.05,
            no_se_reward=-0.25,
            categories=("self_a",),
        ),
    )
    stage3 = StageConfig(
        stage=3,
        stimulus_mix=dict(base_stage1.stimulus_mix),
        entry_condition="stability_detected",
        exit_condition="stability_detected_or_max_trials",
        max_trials=5000,
        conflict=ConflictConfig(
            maintain_reward=-1.0,
            change_reward=1.0,
            no_se_reward=-0.25,
            categories=("self_a",),
        ),
    )
    third = 0.4 / 3
    stage4 = StageConfig(
        stage=4,
        stimulus_mix={
            "real": 0.3,
            "falsa": 0.3,
            "self_a": third,
            "self_b": third,
            "other": third,
        },
        entry_condition="stability_detected",
        exit_condition="stability_detected_or_max_trials",
        max_trials=5000,
        conflict=ConflictConfig(
            maintain_reward=-1.0,
            change_reward=0.5,
            no_se_reward=-0.1,
            categories=("self_a", "self_b", "other"),
        ),
    )
    return [stage2, stage3, stage4]


def condition_progressive_conflict_config(
    seed: int = 0, run_name: str = "self_fly_conflict"
) -> ExperimentConfig:
    """default_config()'s stage 0/1 (self_a only) followed by the three
    progressive-conflict stages from phase_b_conflict_stages(). Stage 4
    introduces SELF_B/OTHER, so experimental_condition is "self_multi" --
    same gate engine.py already uses to train the auxiliary OTHER agent."""
    base = default_config(seed=seed, run_name=run_name)
    stage0, stage1 = base.stages
    return replace(
        base,
        stages=[stage0, stage1, *phase_b_conflict_stages(stage1)],
        experimental_condition="self_multi",
        self_variant=SelfVariantConfig(),
    )
