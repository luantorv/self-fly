import pytest

from self_fly.agent.actions import Action
from self_fly.config.defaults import default_config
from self_fly.environment.stimulus_types import StimulusLabel
from self_fly.rewards.reward_model import RewardModel, UndefinedRewardError


def test_stage0_rewards_match_config():
    config = default_config()
    model = RewardModel(config)
    assert model.compute(StimulusLabel.REAL, Action.REAL, stage=0) == config.reward.correct
    assert model.compute(StimulusLabel.REAL, Action.FALSA, stage=0) == config.reward.incorrect
    assert model.compute(StimulusLabel.FALSA, Action.FALSA, stage=0) == config.reward.correct
    assert model.compute(StimulusLabel.FALSA, Action.REAL, stage=0) == config.reward.incorrect
    assert model.compute(StimulusLabel.REAL, Action.NO_SE, stage=0) == config.reward.dont_know
    assert model.compute(StimulusLabel.FALSA, Action.NO_SE, stage=0) == config.reward.dont_know


def test_stage1_self_stimulus_is_neutral_by_explicit_override():
    config = default_config()
    model = RewardModel(config)
    for action in (Action.REAL, Action.FALSA, Action.NO_SE):
        assert model.compute(StimulusLabel.SELF_A, action, stage=1) == 0.0


def test_undefined_reward_raises_instead_of_guessing():
    config = default_config()
    model = RewardModel(config)
    with pytest.raises(UndefinedRewardError):
        model.compute(StimulusLabel.SELF_A, Action.REAL, stage=0)
