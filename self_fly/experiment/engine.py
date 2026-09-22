from __future__ import annotations

import datetime as dt

import numpy as np

from self_fly.agent.actions import ACTIONS
from self_fly.agent.factory import make_agent
from self_fly.config.conditions import CONDITIONS_REQUIRING_AUXILIARY
from self_fly.config.schema import ExperimentConfig
from self_fly.environment.environment import Environment
from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel
from self_fly.rewards.reward_model import RewardModel
from self_fly.stimuli.control_stimulus import ControlStimulusGenerator
from self_fly.stimuli.generator import StimulusGenerator
from self_fly.stimuli.self_stimulus import SelfStimulusGenerator
from self_fly.stimuli.self_variant import build_self_epsilon, neutral_signature

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
from .validation import build_validation_set, task_accuracy


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
        # Four disjoint streams instead of one. With a single stream the
        # very first draw already diverged between conditions (each has a
        # different label set), so two conditions run on the same seed saw
        # different stimuli from trial 1 and were not comparable trial by
        # trial. Split this way:
        #   rng_calendar     - which trials are experimental slots, and of
        #                      which type. Identical across conditions.
        #   rng_content      - ordinary REAL/FALSA on non-slot trials.
        #                      Identical across conditions.
        #   rng_slot_content - ordinary stimuli used to FILL slots in
        #                      BASELINE/FRESH. Separate so that doing so
        #                      cannot shift rng_content.
        #   rng_action       - action sampling. Diverges between conditions
        #                      as soon as the policies differ, which is
        #                      unavoidable and harmless.
        self.rng_calendar = np.random.default_rng(config.seed + 100)
        self.rng_content = np.random.default_rng(config.seed + 200)
        self.rng_slot_content = np.random.default_rng(config.seed + 300)
        self.rng_action = np.random.default_rng(config.seed + 400)
        self.rng = self.rng_action  # backwards-compatible alias
        self.stimulus_generator = StimulusGenerator(config.stimulus, self.rng_content)
        self.slot_stimulus_generator = StimulusGenerator(config.stimulus, self.rng_slot_content)
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

        self.stability_detector = StabilityDetector(
            config.stability, accuracy_fn=self.task_accuracy
        )
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
        # Only populated for experimental_condition == "self_multi".
        # self_b now holds SELF(eps*), the distance-matched variant.
        self.self_b_stimulus_ground_truth: GroundTruthStimulus | None = None
        self.other_stimulus_ground_truth: GroundTruthStimulus | None = None
        self.other_2_stimulus_ground_truth: GroundTruthStimulus | None = None
        self.other_agent_outcome: StageOutcome | None = None
        self.auxiliary_accuracy: dict[str, float] = {}
        self.epsilon_diagnostics: dict = {}
        self._last_snapshot_name: str | None = None
        self._stage_entry_snapshot_name: str | None = None
        self._pi1_pre_captured = False

        # S_ref and the validation set come from the PROTOCOL seed, not the
        # experimental seed: every seed and every condition is then measured
        # against the same probes, so JS values and accuracies are directly
        # comparable across seeds. (Deriving them from `seed + 1`, as before,
        # gave each seed its own ruler.) Being a separate stream, generating
        # them also never perturbs the experiment's trial sequence.
        self.reference_set = build_reference_set(
            config.stimulus, np.random.default_rng(config.protocol.protocol_seed)
        )
        self.validation_set = build_validation_set(config.stimulus, config.protocol)
        self.temporal_distance_tracker = TemporalDistanceTracker(self.reference_set)

        self.policy_snapshots: dict[str, PolicySnapshot] = {}
        self.stage_events: list[StageEvent] = []
        self._trial_index = 0

    @property
    def current_stage(self) -> int:
        return int(self.stage_machine.current_stage)

    @staticmethod
    def _is_special_exposure(ground_truth: GroundTruthStimulus) -> bool:
        """Anything that is not a plain REAL/FALSA judgement -- whichever
        special category this condition happens to use."""
        return ground_truth.label not in (StimulusLabel.REAL, StimulusLabel.FALSA)

    def special_stimuli_features(self) -> dict:
        """The frozen special stimuli that exist in this run, as raw
        feature vectors. Whether a stimulus was ever PRESENTED to the
        agent is a separate matter: these can be evaluated against any
        snapshot regardless of the condition."""
        candidates = {
            "self_a": self.self_stimulus_ground_truth,
            "self_epsilon": self.self_b_stimulus_ground_truth,
            "other": self.other_stimulus_ground_truth,
            "other_2": self.other_2_stimulus_ground_truth,
            "control": self.control_stimulus_ground_truth,
        }
        return {
            name: np.array(gt.features.to_tuple())
            for name, gt in candidates.items()
            if gt is not None
        }

    def task_accuracy(self) -> float:
        """Accuracy on the fixed held-out REAL/FALSA validation set. Never
        produces reward or an update, and goes through `evaluate()` so it
        leaves a recurrent agent's state untouched."""
        return task_accuracy(self.agent, self.validation_set)

    def step(self) -> Trial:
        stage = self.current_stage
        ground_truth, is_slot = self._sample_stimulus(stage)

        # pi_1_pre must be the policy BEFORE this stimulus touches the agent
        # at all -- before the forward pass (which advances a recurrent
        # agent's state) and before the update. Captured here rather than
        # from the stage machine, which only sees the trial once both have
        # already happened. Separating pre from post is what makes
        # JS(pi_1_pre, pi_1_post) the effect of the first exposure itself,
        # instead of that effect mixed with drift since pi_0.
        #
        # Keyed on the SLOT, not on the stimulus being special, so BASELINE
        # captures the pair at the same moment as every other condition and
        # acts as a time-matched control.
        if not self._pi1_pre_captured and is_slot and stage > 0:
            self._pi1_pre_captured = True
            self._capture_snapshot("pi_1_pre", stage, self._trial_index)

        observable = self.environment.observe(ground_truth)
        features = np.array(observable.features.to_tuple())

        probs = self.agent.probabilities(features)
        action = self.agent.sample(probs, self.rng_action)
        action_idx = ACTIONS.index(action)

        reward = self.reward_model.compute(
            ground_truth.label, action, stage, category_id=ground_truth.category_id
        )
        self.agent.update(features, action_idx, reward, probs)
        policy_js_delta = self.temporal_distance_tracker.step(self.agent)

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
            is_slot=is_slot,
        )
        self._trial_index += 1

        if self.logger is not None:
            self.logger.log_trial(trial)
        if self.metrics is not None:
            self.metrics.update(trial)

        stability_event = self.stability_detector.update(trial, policy_js_delta)
        for stage_event in self.stage_machine.process(trial, stability_event):
            self._handle_stage_event(stage_event)

        return trial

    def run(self, n_trials: int) -> list[Trial]:
        return [self.step() for _ in range(n_trials)]

    _SPECIAL_LABELS = ("self_a", "self_b", "other", "control", "fresh")

    def _sample_stimulus(self, stage: int) -> GroundTruthStimulus:
        """Draw this trial's stimulus against the shared experimental calendar.

        Two draws are taken from `rng_calendar` on EVERY trial, in the same
        order, whatever the condition -- so the calendar stays aligned
        across conditions run with the same seed: slot k lands on the same
        trial index in BASELINE, CONTROL, SELF and OTHER alike. Ordinary
        REAL/FALSA content comes from its own stream for the same reason,
        so non-slot trials are identical across conditions too, and the
        only thing that varies is what fills a slot.
        """
        stage_config = self.stage_machine.current_stage_config()
        mix = stage_config.stimulus_mix
        specials = {k: v for k, v in mix.items() if k in self._SPECIAL_LABELS}

        slot_fraction = stage_config.slot_fraction
        if slot_fraction is None:
            slot_fraction = sum(specials.values())

        slot_draw = self.rng_calendar.random()
        type_draw = self.rng_calendar.random()
        is_slot = slot_draw < slot_fraction

        if is_slot and specials:
            return self._fill_slot(specials, type_draw, stage), True

        # Either a non-slot trial, or a slot in a condition that fills
        # slots with ordinary stimuli (BASELINE). The latter draws from a
        # separate stream so it cannot desynchronise the shared content.
        rng = self.rng_slot_content if is_slot else self.rng_content
        generator = self.slot_stimulus_generator if is_slot else self.stimulus_generator
        chosen = StimulusLabel.REAL if rng.random() < 0.5 else StimulusLabel.FALSA
        return generator.sample_labeled(stage, chosen), is_slot

    def _fill_slot(
        self, specials: dict, type_draw: float, stage: int
    ) -> GroundTruthStimulus:
        names = sorted(specials)
        weights = np.array([specials[n] for n in names], dtype=float)
        cumulative = np.cumsum(weights / weights.sum())
        chosen = names[int(np.searchsorted(cumulative, type_draw))]

        if chosen == StimulusLabel.FRESH.value:
            # The one special category that is NOT frozen: a new
            # in-distribution stimulus each time, relabelled so the
            # experiment can tell it apart while the agent cannot.
            sample = self.slot_stimulus_generator.sample(stage)
            return GroundTruthStimulus(
                stimulus_id=f"fresh_{sample.stimulus_id}",
                label=StimulusLabel.FRESH,
                features=sample.features,
                category_id="fresh",
                stage=stage,
            )

        frozen = {
            StimulusLabel.SELF_A.value: self.self_stimulus_ground_truth,
            StimulusLabel.SELF_B.value: self.self_b_stimulus_ground_truth,
            StimulusLabel.OTHER.value: self.other_stimulus_ground_truth,
            StimulusLabel.CONTROL.value: self.control_stimulus_ground_truth,
        }[chosen]
        assert frozen is not None, (
            f"{chosen} requested by the stimulus mix before it was frozen at pi_0 "
            f"(auxiliary agent outcome: {self.other_agent_outcome})"
        )
        return frozen

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
                if self.config.experimental_condition in CONDITIONS_REQUIRING_AUXILIARY:
                    self._prepare_self_multi_stimuli(event.trial_index)
        elif event.kind == "stage_changed":
            self.stability_detector.reset()
            self._stage_entry_snapshot_name = self._last_snapshot_name

    def _prepare_self_multi_stimuli(self, trial_index: int) -> None:
        """Build OTHER_1, OTHER_2 and SELF(eps*) once pi_0 exists.

        OTHER_2 is not an extra condition: it supplies the null that H1
        has to be read against. "SELF_A differs from OTHER" means nothing
        on its own, because any two independently trained agents differ;
        the question is whether that difference exceeds the spread between
        two agents who are BOTH strangers. Training both here keeps that
        null paired with the same seed.
        """
        variant = self.config.self_variant
        own_signature = self.agent.self_signature(transform=variant.self_a_transform)

        others = {}
        for key, offset in (
            ("other", variant.other_agent_seed_offset),
            ("other_2", variant.second_other_agent_seed_offset),
        ):
            agent, outcome, accuracy = train_auxiliary_agent_to_stability(
                self.config,
                seed=self.config.seed + offset,
                stage0_config=self.config.stages[0],
                max_trials=5000,
            )
            if key == "other":
                self.other_agent_outcome = outcome
            self.auxiliary_accuracy[key] = accuracy

            # An auxiliary agent that did not learn the task carries no
            # policy to speak of, so its signature is not "another agent's
            # policy" in any meaningful sense. Refuse it rather than
            # quietly building OTHER out of noise.
            if accuracy >= self.config.stability.min_accuracy:
                others[key] = self.self_stimulus_generator.generate(
                    agent.self_signature(transform=variant.self_a_transform),
                    stage=1,
                    label=StimulusLabel.OTHER,
                )
            else:
                outcome = None

            event = StageEvent(
                "auxiliary_agent_outcome",
                trial_index,
                {
                    "which": key,
                    "outcome": outcome.value if outcome is not None else "OTHER_INVALID",
                    "accuracy": accuracy,
                },
            )
            self.stage_events.append(event)
            if self.logger is not None:
                self.logger.log_event(event)

        self.other_stimulus_ground_truth = others.get("other")
        self.other_2_stimulus_ground_truth = others.get("other_2")

        if self.other_stimulus_ground_truth is not None:
            own_features = np.array(
                self.self_stimulus_generator.generate(
                    own_signature, stage=1, label=StimulusLabel.SELF_A
                ).features.to_tuple()
            )
            other_features = np.array(
                self.other_stimulus_ground_truth.features.to_tuple()
            )
            self.self_b_stimulus_ground_truth, self.epsilon_diagnostics = build_self_epsilon(
                self.self_stimulus_generator,
                own_signature,
                neutral_signature(len(own_signature), variant, self.config.protocol),
                target_distance=float(np.linalg.norm(own_features - other_features)),
                variant=variant,
                stage=1,
            )

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
        # evaluate(), not probabilities(): capturing a snapshot must not
        # advance a recurrent agent's state -- measuring is not experience.
        evaluated = self.agent.evaluate(
            self.reference_set, conditioning_state=self.agent.recurrent_state()
        )
        reference_probs = {f"ref{i}": tuple(p.tolist()) for i, p in enumerate(evaluated)}

        payload = self.agent.snapshot_payload()
        snapshot = PolicySnapshot(
            name=name,
            stage=stage,
            trial_index=trial_index,
            weights=payload["weights"],
            bias=payload["bias"],
            reference_action_probs=reference_probs,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            agent_state=self.agent.full_state(),
        )
        self.policy_snapshots[name] = snapshot
        self._last_snapshot_name = name
        if self.logger is not None:
            self.logger.log_snapshot(snapshot)
        return snapshot
