from __future__ import annotations

import datetime as dt

import numpy as np

from self_fly.agent.actions import ACTIONS
from self_fly.agent.learner import REINFORCEUpdate
from self_fly.agent.policy import LinearSoftmaxPolicy
from self_fly.agent.state import AgentInternalState
from self_fly.config.schema import ExperimentConfig
from self_fly.environment.environment import Environment
from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel
from self_fly.rewards.reward_model import RewardModel
from self_fly.stimuli.generator import StimulusGenerator
from self_fly.stimuli.self_stimulus import SelfStimulusGenerator

from .policy_distance import build_reference_set
from .stability import StabilityDetector
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
        self.agent = AgentInternalState(
            policy=LinearSoftmaxPolicy(config.learner),
            learner=REINFORCEUpdate(config.learner),
        )
        self.reward_model = RewardModel(config)
        self.logger = logger
        self.metrics = metrics

        self.stability_detector = StabilityDetector(config.stability)
        self.stage_machine = StageMachine(config.stages)
        self.self_stimulus_generator = SelfStimulusGenerator()
        self.self_stimulus_ground_truth: GroundTruthStimulus | None = None

        # Separate RNG stream: S_ref generation never perturbs the main
        # experiment sequence, so the same seed always reproduces it.
        self.reference_set = build_reference_set(
            config.stimulus, np.random.default_rng(config.seed + 1)
        )

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

        probs = self.agent.policy.probabilities(features)
        action = self.agent.policy.sample(probs, self.rng)
        action_idx = ACTIONS.index(action)

        reward = self.reward_model.compute(ground_truth.label, action, stage)
        self.agent.learner.update(self.agent.policy, features, action_idx, reward, probs)

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

        return self.stimulus_generator.sample_labeled(stage, StimulusLabel(chosen))

    def _handle_stage_event(self, event: StageEvent) -> None:
        self.stage_events.append(event)
        if self.logger is not None:
            self.logger.log_event(event)

        if event.kind == "capture_snapshot":
            name = event.payload["name"]
            stage = event.payload["stage"]
            self._capture_snapshot(name, stage, event.trial_index)
            if name == "pi_0":
                self.self_stimulus_ground_truth = self.self_stimulus_generator.generate(
                    self.agent.policy, stage=1
                )
        elif event.kind == "stage_changed":
            self.stability_detector.reset()

    def _capture_snapshot(self, name: str, stage: int, trial_index: int) -> PolicySnapshot:
        reference_probs = {}
        for i, features in enumerate(self.reference_set):
            probs = self.agent.policy.probabilities(features)
            reference_probs[f"ref{i}"] = tuple(probs.tolist())

        snapshot = PolicySnapshot(
            name=name,
            stage=stage,
            trial_index=trial_index,
            weights=self.agent.policy.W.tolist(),
            bias=self.agent.policy.b.tolist(),
            reference_action_probs=reference_probs,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self.policy_snapshots[name] = snapshot
        if self.logger is not None:
            self.logger.log_snapshot(snapshot)
        return snapshot
