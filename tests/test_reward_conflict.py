from self_fly.agent.actions import Action
from self_fly.config.schema import ConflictConfig, ExperimentConfig, RewardConfig, StageConfig
from self_fly.environment.stimulus_types import StimulusLabel
from self_fly.rewards.reward_model import RewardModel, UndefinedRewardError


def _config_with_conflict(conflict: ConflictConfig) -> ExperimentConfig:
    stage = StageConfig(
        stage=1,
        stimulus_mix={"real": 0.5, "falsa": 0.5},
        conflict=conflict,
    )
    return ExperimentConfig(reward=RewardConfig(), stages=[stage])


def test_first_exposure_uses_first_exposure_reward():
    conflict = ConflictConfig(
        maintain_reward=1.0, change_reward=-1.0, no_se_reward=-0.25, first_exposure_reward=0.42
    )
    model = RewardModel(_config_with_conflict(conflict))

    reward = model.compute(StimulusLabel.SELF_A, Action.REAL, stage=1, category_id="self_a")

    assert reward == 0.42


def test_repeating_the_same_action_is_rewarded_as_maintain():
    conflict = ConflictConfig(maintain_reward=1.0, change_reward=-1.0, no_se_reward=-0.25)
    model = RewardModel(_config_with_conflict(conflict))
    model.compute(StimulusLabel.SELF_A, Action.REAL, stage=1, category_id="self_a")

    reward = model.compute(StimulusLabel.SELF_A, Action.REAL, stage=1, category_id="self_a")

    assert reward == 1.0


def test_switching_action_is_rewarded_as_change():
    conflict = ConflictConfig(maintain_reward=1.0, change_reward=-1.0, no_se_reward=-0.25)
    model = RewardModel(_config_with_conflict(conflict))
    model.compute(StimulusLabel.SELF_A, Action.REAL, stage=1, category_id="self_a")

    reward = model.compute(StimulusLabel.SELF_A, Action.FALSA, stage=1, category_id="self_a")

    assert reward == -1.0


def test_no_se_uses_its_own_reward_and_never_updates_history():
    conflict = ConflictConfig(maintain_reward=1.0, change_reward=-1.0, no_se_reward=-0.99)
    model = RewardModel(_config_with_conflict(conflict))
    model.compute(StimulusLabel.SELF_A, Action.REAL, stage=1, category_id="self_a")

    reward = model.compute(StimulusLabel.SELF_A, Action.NO_SE, stage=1, category_id="self_a")
    assert reward == -0.99

    # History untouched by NO_SE: the next REAL is still judged against the earlier REAL.
    reward = model.compute(StimulusLabel.SELF_A, Action.REAL, stage=1, category_id="self_a")
    assert reward == 1.0


def test_categories_are_tracked_independently():
    conflict = ConflictConfig(
        maintain_reward=1.0, change_reward=-1.0, no_se_reward=-0.25, categories=("self_a", "self_b")
    )
    model = RewardModel(_config_with_conflict(conflict))
    model.compute(StimulusLabel.SELF_A, Action.REAL, stage=1, category_id="self_a")

    # self_b has no history of its own yet -- first exposure, independent of self_a's.
    reward = model.compute(StimulusLabel.SELF_B, Action.FALSA, stage=1, category_id="self_b")

    assert reward == 0.0  # ConflictConfig.first_exposure_reward default


def test_labels_outside_conflict_categories_use_base_correctness_rules():
    conflict = ConflictConfig(
        maintain_reward=1.0, change_reward=-1.0, no_se_reward=-0.25, categories=("self_a",)
    )
    model = RewardModel(_config_with_conflict(conflict))

    reward = model.compute(StimulusLabel.REAL, Action.REAL, stage=1, category_id="real_c00")

    assert reward == 1.0  # RewardConfig.correct default


def test_label_outside_categories_and_without_overrides_still_fails_loud():
    conflict = ConflictConfig(
        maintain_reward=1.0, change_reward=-1.0, no_se_reward=-0.25, categories=("self_a",)
    )
    model = RewardModel(_config_with_conflict(conflict))

    try:
        model.compute(StimulusLabel.SELF_B, Action.REAL, stage=1, category_id="self_b")
        raise AssertionError("expected UndefinedRewardError")
    except UndefinedRewardError:
        pass
