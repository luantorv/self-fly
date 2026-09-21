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
        # Mutable by design (unlike the rest of this class): a maintain/
        # change reward needs to remember the last non-NO_SE action taken
        # toward each conflict category. Keyed by category_id (falling back
        # to label.value if the caller doesn't have one), never shared
        # across RewardModel instances -- each ExperimentEngine owns its own.
        self._last_action_by_category: dict[str, Action] = {}

    def compute(
        self,
        label: StimulusLabel,
        action: Action,
        stage: int,
        category_id: str | None = None,
    ) -> float:
        stage_cfg = self._stage_by_number.get(stage)

        if stage_cfg is not None and stage_cfg.conflict is not None and label.value in stage_cfg.conflict.categories:
            return self._compute_conflict_reward(stage_cfg.conflict, label, action, category_id)

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

    def _compute_conflict_reward(
        self,
        conflict,
        label: StimulusLabel,
        action: Action,
        category_id: str | None,
    ) -> float:
        key = category_id or label.value

        if action == Action.NO_SE:
            return conflict.no_se_reward

        previous = self._last_action_by_category.get(key)
        reward = (
            conflict.first_exposure_reward
            if previous is None
            else (conflict.maintain_reward if action == previous else conflict.change_reward)
        )
        self._last_action_by_category[key] = action
        return reward
