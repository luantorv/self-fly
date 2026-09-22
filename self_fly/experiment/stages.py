from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from self_fly.config.schema import StageConfig

from .stability import StabilityDetector, StabilityEvent
from .stage_outcome import StageOutcome
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


_CONFLICT_STAGE_SNAPSHOT_NAMES = {
    Stage.STAGE_2_MILD_CONFLICT: "pi_3",
    Stage.STAGE_3_STRONG_CONFLICT: "pi_4",
    Stage.STAGE_4_THREE_WAY_CONFLICT: "pi_5",
}


class StageMachine:
    """Phase A drives only stage 0 -> stage 1 -> finished:

      stage 0: normal REAL/FALSA learning until the policy is stable
               -> capture pi_0, freeze the self stimulus, move to stage 1.
      stage 1: self stimulus mixed in, no special consequences
               -> capture pi_1 on first exposure; capture pi_2 and finish
                  once stable again (or a trial budget is exhausted).

    Stages 2-4 (progressive conflict) are declared in `Stage` and are
    driven by `_process_conflict_stage`, shared across all three -- they
    differ only in which StageConfig.conflict rewards are active, not in
    how entry/exit/snapshot naming works. Whether a stage's conclusion
    advances into the next configured stage or ends the experiment is
    decided generically (see `_exit_or_advance`): a StageConfig for
    stage N+1 in the same run means stage N's conclusion advances into it,
    its absence means the experiment finishes there -- this is what lets
    default_config()'s stage 1 keep finishing exactly as before while a
    conflict-augmented config continues into stage 2.
    """

    def __init__(
        self,
        stage_configs: list[StageConfig],
        stability_detector: StabilityDetector | None = None,
    ):
        self._config_by_stage = {c.stage: c for c in stage_configs}
        self.current_stage = Stage.STAGE_0_LEARNING
        self.finished = False
        self._self_exposed = False
        self._stage_entry_trial_index = 0
        # Read (never mutated) only to tell UNSTABLE apart from TIMEOUT when
        # a stage exits via max_trials instead of real stability.
        self._stability_detector = stability_detector

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
        elif self.current_stage in _CONFLICT_STAGE_SNAPSHOT_NAMES:
            events.extend(self._process_conflict_stage(trial, stability_event))

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

        # Keyed on the experimental SLOT rather than on the stimulus being
        # special. In BASELINE the slots carry ordinary REAL/FALSA, and it
        # still has to reach pi_1 and be able to conclude stage 1 by
        # stability -- otherwise the one condition that serves as the null
        # would be structurally incapable of producing STABLE, and its
        # outcomes could not be compared with anything.
        if not self._self_exposed and trial.is_slot:
            self._self_exposed = True
            events.append(
                StageEvent(
                    "capture_snapshot",
                    trial.trial_index,
                    # "post": the policy AFTER the update caused by this first
                    # exposure. Its counterpart pi_1_pre is captured by the
                    # engine before the stimulus reaches the agent -- the pair
                    # is what isolates the effect of the exposure from drift.
                    {"name": "pi_1_post", "stage": int(Stage.STAGE_1_SELF_INTRODUCED)},
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
            outcome = self._resolve_outcome(hit_stability)
            snapshot_name = "pi_2" if outcome == StageOutcome.STABLE else f"pi_2_{outcome.value}"
            events.extend(
                self._exit_or_advance(trial, outcome, snapshot_name, trials_in_stage)
            )

        return events

    def _process_conflict_stage(
        self, trial: Trial, stability_event: Optional[StabilityEvent]
    ) -> list[StageEvent]:
        stage_cfg = self.current_stage_config()
        trials_in_stage = trial.trial_index - self._stage_entry_trial_index + 1
        hit_max_trials = (
            stage_cfg.max_trials is not None and trials_in_stage >= stage_cfg.max_trials
        )
        hit_stability = stability_event is not None

        should_exit = {
            "stability_detected": hit_stability,
            "max_trials": hit_max_trials,
            "stability_detected_or_max_trials": hit_stability or hit_max_trials,
        }.get(stage_cfg.exit_condition, False)

        if not should_exit:
            return []

        outcome = self._resolve_outcome(hit_stability)
        base_name = _CONFLICT_STAGE_SNAPSHOT_NAMES[self.current_stage]
        snapshot_name = base_name if outcome == StageOutcome.STABLE else f"{base_name}_{outcome.value}"
        return self._exit_or_advance(trial, outcome, snapshot_name, trials_in_stage)

    def _exit_or_advance(
        self,
        trial: Trial,
        outcome: StageOutcome,
        snapshot_name: str,
        trials_in_stage: int,
    ) -> list[StageEvent]:
        """Shared tail for every stage that can conclude (1 through 4):
        capture its snapshot, report the outcome, then either advance into
        the next configured stage or finish -- whichever the run's own
        StageConfig list actually provides, decided generically so adding
        stage N+1's config is the only thing needed to chain stages."""
        concluding_stage = int(self.current_stage)
        events = [
            StageEvent(
                "capture_snapshot",
                trial.trial_index,
                {"name": snapshot_name, "stage": concluding_stage},
            ),
            StageEvent(
                "stage_outcome",
                trial.trial_index,
                {
                    "stage": concluding_stage,
                    "outcome": outcome.value,
                    "snapshot_name": snapshot_name,
                    "trials_in_stage": trials_in_stage,
                },
            ),
        ]

        next_stage_value = concluding_stage + 1
        if next_stage_value in self._config_by_stage:
            events.append(
                StageEvent(
                    "stage_changed",
                    trial.trial_index,
                    {"from_stage": concluding_stage, "to_stage": next_stage_value},
                )
            )
            self.current_stage = Stage(next_stage_value)
            self._stage_entry_trial_index = trial.trial_index + 1
        else:
            events.append(StageEvent("experiment_finished", trial.trial_index, {}))
            self.finished = True

        return events

    def _resolve_outcome(self, hit_stability: bool) -> StageOutcome:
        if hit_stability:
            return StageOutcome.STABLE
        if self._stability_detector is not None and self._stability_detector.max_consecutive_stable_observed > 0:
            return StageOutcome.UNSTABLE
        return StageOutcome.TIMEOUT
