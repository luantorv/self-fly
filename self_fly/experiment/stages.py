from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from self_fly.config.schema import StageConfig

from .stability import StabilityEvent
from .types import Trial


class Stage(IntEnum):
    STAGE_0_LEARNING = 0
    STAGE_1_SELF_INTRODUCED = 1
    STAGE_2_MILD_CONFLICT = 2  # Phase B
    STAGE_3_STRONG_CONFLICT = 3  # Phase B
    STAGE_4_THREE_WAY_CONFLICT = 4  # Phase B


@dataclass
class StageEvent:
    kind: str  # "capture_snapshot" | "stage_changed" | "experiment_finished"
    trial_index: int
    payload: dict = field(default_factory=dict)


class StageMachine:
    """Phase A drives only stage 0 -> stage 1 -> finished:

      stage 0: normal REAL/FALSA learning until the policy is stable
               -> capture pi_0, freeze the self stimulus, move to stage 1.
      stage 1: self stimulus mixed in, no special consequences
               -> capture pi_1 on first exposure; capture pi_2 and finish
                  once stable again (or a trial budget is exhausted).

    Stages 2-4 (progressive conflict) are declared in `Stage` and can be
    added as StageConfig entries with populated stimulus_mix/reward
    overrides without touching this class's stage-0/1 logic -- only the
    `process` method needs a new branch per added stage.
    """

    def __init__(self, stage_configs: list[StageConfig]):
        self._config_by_stage = {c.stage: c for c in stage_configs}
        self.current_stage = Stage.STAGE_0_LEARNING
        self.finished = False
        self._self_exposed = False
        self._stage_entry_trial_index = 0

    def current_stage_config(self) -> StageConfig:
        return self._config_by_stage[int(self.current_stage)]

    def stimulus_mix(self) -> dict[str, float]:
        return self.current_stage_config().stimulus_mix

    def process(self, trial: Trial, stability_event: Optional[StabilityEvent]) -> list[StageEvent]:
        events: list[StageEvent] = []
        if self.finished:
            return events

        if self.current_stage == Stage.STAGE_0_LEARNING:
            events.extend(self._process_stage_0(trial, stability_event))
        elif self.current_stage == Stage.STAGE_1_SELF_INTRODUCED:
            events.extend(self._process_stage_1(trial, stability_event))

        return events

    def _process_stage_0(
        self, trial: Trial, stability_event: Optional[StabilityEvent]
    ) -> list[StageEvent]:
        if stability_event is None:
            return []

        events = [
            StageEvent(
                "capture_snapshot",
                trial.trial_index,
                {"name": "pi_0", "stage": int(Stage.STAGE_0_LEARNING)},
            ),
            StageEvent(
                "stage_changed",
                trial.trial_index,
                {
                    "from_stage": int(Stage.STAGE_0_LEARNING),
                    "to_stage": int(Stage.STAGE_1_SELF_INTRODUCED),
                },
            ),
        ]
        self.current_stage = Stage.STAGE_1_SELF_INTRODUCED
        self._stage_entry_trial_index = trial.trial_index + 1
        return events

    def _process_stage_1(
        self, trial: Trial, stability_event: Optional[StabilityEvent]
    ) -> list[StageEvent]:
        events: list[StageEvent] = []

        if not self._self_exposed and trial.ground_truth_label == "self_a":
            self._self_exposed = True
            events.append(
                StageEvent(
                    "capture_snapshot",
                    trial.trial_index,
                    {"name": "pi_1", "stage": int(Stage.STAGE_1_SELF_INTRODUCED)},
                )
            )

        stage_cfg = self.current_stage_config()
        trials_in_stage = trial.trial_index - self._stage_entry_trial_index + 1
        hit_max_trials = (
            stage_cfg.max_trials is not None and trials_in_stage >= stage_cfg.max_trials
        )
        hit_stability = stability_event is not None and self._self_exposed

        should_exit = {
            "stability_detected": hit_stability,
            "max_trials": hit_max_trials,
            "stability_detected_or_max_trials": hit_stability or hit_max_trials,
        }.get(stage_cfg.exit_condition, False)

        if should_exit:
            events.append(
                StageEvent(
                    "capture_snapshot",
                    trial.trial_index,
                    {"name": "pi_2", "stage": int(Stage.STAGE_1_SELF_INTRODUCED)},
                )
            )
            events.append(StageEvent("experiment_finished", trial.trial_index, {}))
            self.finished = True

        return events
