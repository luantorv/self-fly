from __future__ import annotations

import datetime as dt

import numpy as np

from self_fly.agent.actions import ACTIONS
from self_fly.agent.factory import make_agent
from self_fly.config.schema import ExperimentConfig
from self_fly.environment.environment import Environment
from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel
from self_fly.rewards.reward_model import RewardModel
from self_fly.stimuli.control_stimulus import ControlStimulusGenerator
from self_fly.stimuli.generator import StimulusGenerator
from self_fly.stimuli.self_stimulus import SelfStimulusGenerator

from .auxiliary_agent import train_auxiliary_agent_to_stability
from .entropy import mean_policy_entropy
from .policy_distance import (
    TemporalDistanceTracker,
    build_reference_set,
    policy_distance_from_snapshots,
)
from .stability import StabilityDetector
from .stage_outcome import StageOutcome
from .stages import StageEvent, StageMachine
from .types import PolicySnapshot, Trial


class ExperimentEngine:
    """The single atomic step() used by both the headless CLI and the GUI.

    Owns the full pipeline: stimulus sampling (respecting the current
    stage's mix, including the frozen self stimulus once it exists) ->
    environment.observe() boundary -> policy -> reward -> learning update ->
    logging/metrics -> stability detection -> stage transitions and
    pi_0/pi_1/pi_2 snapshot capture.
    """

    def __init__(self, config: ExperimentConfig, logger=None, metrics=None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.stimulus_generator = StimulusGenerator(config.stimulus, self.rng)
        self.environment = Environment()
        # seed+3: the agent's own internal RNG needs (only used by
        # ConnectomeInspiredAgent's fixed reservoir init) get their own
        # disjoint stream too, same pattern as S_ref (+1) and the control
        # stimulus (+2) -- so which agent_type runs never perturbs the
        # main trial sequence.
        self.agent = make_agent(
            config.agent_type, config.learner, config.connectome, seed=config.seed + 3
        )
        self.reward_model = RewardModel(config)
        self.logger = logger
        self.metrics = metrics

        self.stability_detector = StabilityDetector(config.stability)
        self.stage_machine = StageMachine(config.stages, stability_detector=self.stability_detector)
        self.self_stimulus_generator = SelfStimulusGenerator()
        self.self_stimulus_ground_truth: GroundTruthStimulus | None = None
        # Own RNG stream (seed+2): generating the control stimulus must
        # never perturb the main experiment's trial sequence, the same way
        # S_ref's seed+1 stream doesn't either.
        self.control_stimulus_generator = ControlStimulusGenerator(
            config.stimulus, np.random.default_rng(config.seed + 2)
        )
        self.control_stimulus_ground_truth: GroundTruthStimulus | None = None
        # Only populated for experimental_condition == "self_multi" (block 7).
        self.self_b_stimulus_ground_truth: GroundTruthStimulus | None = None
        self.other_stimulus_ground_truth: GroundTruthStimulus | None = None
        self.other_agent_outcome: StageOutcome | None = None
        self._last_snapshot_name: str | None = None
        self._stage_entry_snapshot_name: str | None = None

        # Separate RNG stream: S_ref generation never perturbs the main
        # experiment sequence, so the same seed always reproduces it.
        self.reference_set = build_reference_set(
            config.stimulus, np.random.default_rng(config.seed + 1)
        )
        self.temporal_distance_tracker = TemporalDistanceTracker(self.reference_set)

        self.policy_snapshots: dict[str, PolicySnapshot] = {}
        self.stage_events: list[StageEvent] = []
        self._trial_index = 0

    @property
    def current_stage(self) -> int:
        return int(self.stage_machine.current_stage)

    def step(self) -> Trial:
        stage = self.current_stage
        ground_truth = self._sample_stimulus(stage)
        observable = self.environment.observe(ground_truth)
        features = np.array(observable.features.to_tuple())

        probs = self.agent.probabilities(features)
        action = self.agent.sample(probs, self.rng)
        action_idx = ACTIONS.index(action)

        reward = self.reward_model.compute(
            ground_truth.label, action, stage, category_id=ground_truth.category_id
        )
        self.agent.update(features, action_idx, reward, probs)
        policy_js_delta = self.temporal_distance_tracker.step(self.agent.probabilities)

        trial = Trial(
            trial_index=self._trial_index,
            stage=stage,
            stimulus_id=ground_truth.stimulus_id,
            category_id=ground_truth.category_id,
            features=observable.features.to_tuple(),
            action=action.value,
            action_probs=tuple(probs.tolist()),
            reward=reward,
            ground_truth_label=ground_truth.label.value,
            policy_weight_norm=self.agent.weight_norm_scalar(),
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            policy_js_delta=policy_js_delta,
        )
        self._trial_index += 1

        if self.logger is not None:
            self.logger.log_trial(trial)
        if self.metrics is not None:
            self.metrics.update(trial)

        stability_event = self.stability_detector.update(trial)
        for stage_event in self.stage_machine.process(trial, stability_event):
            self._handle_stage_event(stage_event)

        return trial

    def run(self, n_trials: int) -> list[Trial]:
        return [self.step() for _ in range(n_trials)]

    def _sample_stimulus(self, stage: int) -> GroundTruthStimulus:
        stimulus_mix = self.stage_machine.stimulus_mix()
        labels = list(stimulus_mix.keys())
        weights = np.array(list(stimulus_mix.values()), dtype=float)
        weights = weights / weights.sum()
        chosen = self.rng.choice(labels, p=weights)

        if chosen == StimulusLabel.SELF_A.value:
            assert self.self_stimulus_ground_truth is not None, (
                "self_a requested by stimulus_mix before pi_0 froze the self stimulus"
            )
            return self.self_stimulus_ground_truth

        if chosen == StimulusLabel.CONTROL.value:
            assert self.control_stimulus_ground_truth is not None, (
                "control requested by stimulus_mix before pi_0 froze the control stimulus"
            )
            return self.control_stimulus_ground_truth

        if chosen == StimulusLabel.SELF_B.value:
            assert self.self_b_stimulus_ground_truth is not None, (
                "self_b requested by stimulus_mix before pi_0 froze it"
            )
            return self.self_b_stimulus_ground_truth

        if chosen == StimulusLabel.OTHER.value:
            assert self.other_stimulus_ground_truth is not None, (
                "other requested by stimulus_mix, but the auxiliary agent never "
                f"reached StageOutcome.STABLE (outcome={self.other_agent_outcome})"
            )
            return self.other_stimulus_ground_truth

        return self.stimulus_generator.sample_labeled(stage, StimulusLabel(chosen))

    def _handle_stage_event(self, event: StageEvent) -> None:
        if event.kind == "stage_outcome":
            # Complete the payload before it is logged, not after: there is
            # only one events.jsonl line per event, so this is the only
            # chance to attach js_to_stage_entry.
            self._enrich_stage_outcome_payload(event)

        self.stage_events.append(event)
        if self.logger is not None:
            self.logger.log_event(event)

        if event.kind == "capture_snapshot":
            name = event.payload["name"]
            stage = event.payload["stage"]
            self._capture_snapshot(name, stage, event.trial_index)
            if name == "pi_0":
                self.self_stimulus_ground_truth = self.self_stimulus_generator.generate(
                    self.agent.self_signature(transform=self.config.self_variant.self_a_transform),
                    stage=1,
                    label=StimulusLabel.SELF_A,
                )
                self.control_stimulus_ground_truth = self.control_stimulus_generator.generate(stage=1)
                if self.config.experimental_condition == "self_multi":
                    self._prepare_self_multi_stimuli(event.trial_index)
        elif event.kind == "stage_changed":
            self.stability_detector.reset()
            self._stage_entry_snapshot_name = self._last_snapshot_name

    def _prepare_self_multi_stimuli(self, trial_index: int) -> None:
        self.self_b_stimulus_ground_truth = self.self_stimulus_generator.generate(
            self.agent.self_signature(transform=self.config.self_variant.self_b_transform),
            stage=1,
            label=StimulusLabel.SELF_B,
        )

        other_agent, outcome = train_auxiliary_agent_to_stability(
            self.config,
            seed=self.config.seed + self.config.self_variant.other_agent_seed_offset,
            stage0_config=self.config.stages[0],
            max_trials=5000,
        )
        self.other_agent_outcome = outcome
        if outcome == StageOutcome.STABLE:
            self.other_stimulus_ground_truth = self.self_stimulus_generator.generate(
                other_agent.self_signature(transform="l2_column_norm"), stage=1, label=StimulusLabel.OTHER
            )

        # Not stage_machine-driven, so it bypasses _handle_stage_event's
        # dispatch -- logged directly so a non-STABLE auxiliary agent is
        # visible in events.jsonl, never silently absorbed.
        outcome_event = StageEvent(
            "auxiliary_agent_outcome",
            trial_index,
            {"label": "other", "outcome": outcome.value},
        )
        self.stage_events.append(outcome_event)
        if self.logger is not None:
            self.logger.log_event(outcome_event)

    def _enrich_stage_outcome_payload(self, event: StageEvent) -> None:
        outcome_snapshot = self.policy_snapshots.get(event.payload.get("snapshot_name"))
        entry_snapshot = (
            self.policy_snapshots.get(self._stage_entry_snapshot_name)
            if self._stage_entry_snapshot_name
            else None
        )
        if outcome_snapshot is not None and entry_snapshot is not None:
            event.payload["js_to_stage_entry"] = policy_distance_from_snapshots(
                outcome_snapshot, entry_snapshot
            )["js"]
        else:
            event.payload["js_to_stage_entry"] = None
        event.payload["entropy"] = (
            mean_policy_entropy(outcome_snapshot) if outcome_snapshot is not None else None
        )

    def _capture_snapshot(self, name: str, stage: int, trial_index: int) -> PolicySnapshot:
        reference_probs = {}
        for i, features in enumerate(self.reference_set):
            probs = self.agent.probabilities(features)
            reference_probs[f"ref{i}"] = tuple(probs.tolist())

        payload = self.agent.snapshot_payload()
        snapshot = PolicySnapshot(
            name=name,
            stage=stage,
            trial_index=trial_index,
            weights=payload["weights"],
            bias=payload["bias"],
            reference_action_probs=reference_probs,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self.policy_snapshots[name] = snapshot
        self._last_snapshot_name = name
        if self.logger is not None:
            self.logger.log_snapshot(snapshot)
        return snapshot
