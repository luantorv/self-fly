from __future__ import annotations

from self_fly.agent.actions import Action
from self_fly.config.schema import ExperimentConfig
from self_fly.environment.stimulus_types import StimulusLabel


class UndefinedRewardError(ValueError):
    pass


class RewardModel:
    """(label, action, stage) -> reward.

    Base REAL/FALSA correctness rules come from RewardConfig. Anything that
    isn't a plain REAL/FALSA judgement (e.g. a self-referential stimulus)
    must be defined explicitly via reward_overrides on the active stage's
    config -- there is no implicit fallback for those, by design, so every
    stage's reward structure is fully visible in its config.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._stage_by_number = {s.stage: s for s in config.stages}

    def compute(self, label: StimulusLabel, action: Action, stage: int) -> float:
        stage_cfg = self._stage_by_number.get(stage)
        if stage_cfg is not None:
            key = f"{label.value}|{action.value}"
            if key in stage_cfg.reward_overrides:
                return stage_cfg.reward_overrides[key]

        if action == Action.NO_SE:
            return self.config.reward.dont_know

        if label in (StimulusLabel.REAL, StimulusLabel.FALSA):
            is_correct = action.value == label.value.upper()
            return self.config.reward.correct if is_correct else self.config.reward.incorrect

        raise UndefinedRewardError(
            f"No reward rule for label={label.value} action={action.value} stage={stage}. "
            "Add an explicit reward_overrides entry in the stage config."
        )
