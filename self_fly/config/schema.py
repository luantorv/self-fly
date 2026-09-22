from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    incorrect: float = -1.0
    dont_know: float = -0.25


@dataclass(frozen=True)
class StabilityConfig:
    """Stability = task competence AND a settled policy, over consecutive
    windows. See experiment/stability.py for why the previous
    reward-coefficient-of-variation criterion was removed.

    `min_accuracy`: the Bayes ceiling of this task at noise_std=0.35 is
    ~80.3%; a trained BaselineAgent reaches ~79% (argmax). 0.70 therefore
    demands real competence while leaving ~9 points of headroom.

    `max_median_policy_delta`: a converged policy shows median D_t of
    ~0.0005-0.001. Set to 0.002 rather than the initially proposed 0.01,
    which was ~10-20x too permissive -- individual windows cleared 0.01
    as early as trial ~100, while still learning. Both values are
    experimental hypotheses to be fixed in calibration and then held
    constant across conditions.
    """

    window_size: int = 100
    min_accuracy: float = 0.70
    max_median_policy_delta: float = 0.002
    consecutive_windows_required: int = 3


@dataclass(frozen=True)
class StimulusConfig:
    signal_feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "shape_roundness": 1.0,
            "color_hue": -1.0,
            "texture_density": 0.8,
        }
    )
    noise_std: float = 0.35
    category_grid_size: int = 3
    reference_set_size: int = 50


@dataclass(frozen=True)
class LearnerConfig:
    learning_rate: float = 0.05
    baseline_momentum: float = 0.99
    initial_weight_scale: float = 0.0


@dataclass(frozen=True)
class ProtocolConfig:
    """Fixed across every run of the protocol, independent of the
    experimental seed.

    `protocol_seed` generates S_ref and the task validation set ONCE for
    the whole protocol, so that JS values and accuracies from different
    experimental seeds are measured against the same probes and are
    therefore directly comparable. (Deriving them per-seed, as before,
    silently made cross-seed averages compare different rulers.)
    """

    protocol_seed: int = 20260101
    validation_set_size: int = 2000


@dataclass(frozen=True)
class ConnectomeConfig:
    """ConnectomeInspiredAgent only.

    `train_sensory` is what makes the agent's own signature depend on what
    it learned rather than on its random initialization -- with it off,
    SELF_A would be derived from a frozen random matrix and the whole
    self-referential premise would not hold for this architecture.

    `init_scheme="xavier"` scales each layer by 1/sqrt(fan_in). The prior
    fixed 0.05 scale drove the hidden state to ||h||~0.006, which made the
    logits ~1e-4 and the gradient (proportional to h) negligible -- the
    agent could not learn at all. "fixed_scale" reproduces that older
    behaviour via `initial_weight_scale`, kept only for comparison.

    Note: unlike BaselineAgent (whose weights start at exactly zero, so
    its initial policy is exactly uniform), ActionValuation must start
    non-zero or the gradient flowing back into the sensory layer would be
    identically zero. The initial policy is therefore near-uniform, not
    uniform.
    """

    hidden_dim: int = 8
    # Calibrated (protocol §33) over leak x hidden_dim x learning_rate.
    # recurrent_leak=0.1 was unusable: only 10% of the current stimulus
    # entered the hidden state, which both starved the readout of the
    # information the label actually depends on AND scaled the sensory
    # gradient by the same factor. 6 of 7 seeds collapsed onto a constant
    # action (accuracy = the base rate of one class). At 0.6: 7/7 seeds
    # reach 75.5-79.7% (mean 77.8%) against a Bayes ceiling of 80.3%.
    #
    # Measured caveat, to report rather than bury: freezing the hidden
    # state at zero gives 78.5% -- slightly BETTER than the live state's
    # 77.9%. The recurrent machinery is real but contributes nothing on
    # this task, which is i.i.d. and therefore has no history to exploit.
    recurrent_leak: float = 0.6
    initial_weight_scale: float = 0.05
    init_scheme: str = "xavier"
    learning_rate: float = 0.02
    baseline_momentum: float = 0.99
    train_sensory: bool = True
    train_recurrent: bool = False


@dataclass(frozen=True)
class SelfVariantConfig:
    """SELF_A's transform, plus the SELF(eps) distance-matched control.

    SELF_B (the L1-vs-L2 variant) is retired as a scientific condition:
    measured correlation between the two normalized signatures is >0.9995
    and the resulting stimuli sit 0.054 apart while SELF_A and OTHER sit
    2.44 apart, so "SELF_A vs SELF_B" could only ever confirm itself.

    SELF(eps) replaces it. It interpolates the agent's own signature
    toward a NEUTRAL signature -- deliberately not toward OTHER's. The
    protocol originally specified interpolating toward s_other with
    SELF(1)=OTHER, but then the distance-matching target
    d(SELF_A, SELF(eps*)) = d(SELF_A, OTHER) is solved exactly by eps*=1,
    at which point SELF(eps*) IS OTHER and the control collapses onto the
    condition it was meant to control for. Interpolating toward a neutral
    signature keeps the matching well-posed: SELF(eps*) ends up as far
    from SELF_A as OTHER is, while not being derived from another agent,
    which isolates authorship at constant distance.
    """

    self_a_transform: str = "l2_column_norm"
    # Neutral interpolation target, drawn from the protocol seed so it is
    # identical across seeds, conditions and architectures.
    neutral_signature_seed_offset: int = 7_000
    # Tolerance on |d(SELF_A,SELF(eps*)) - d(SELF_A,OTHER)| / d(SELF_A,OTHER).
    epsilon_match_tolerance: float = 0.02
    other_agent_seed_offset: int = 10_000
    second_other_agent_seed_offset: int = 20_000


@dataclass(frozen=True)
class ControlStimulusConfig:
    # Placeholder for now (block 6 needs only "frozen once, in-distribution
    # features" -- see ControlStimulusGenerator). Left as its own dataclass,
    # not inlined, so later variants (e.g. re-sampled per trial) have
    # somewhere to add fields without touching ExperimentConfig again.
    freeze_at_pi0: bool = True


@dataclass(frozen=True)
class ConflictConfig:
    """Reward for maintaining vs. changing the last non-NO_SE action taken
    toward each of `categories` (tracked per category_id, not globally --
    see RewardModel._last_action_by_category). Every field is an
    experimental hypothesis about how sharply to reward/punish
    consistency, not a "correct" value -- report alongside results."""

    maintain_reward: float
    change_reward: float
    no_se_reward: float
    first_exposure_reward: float = 0.0
    categories: tuple[str, ...] = ("self_a",)


@dataclass(frozen=True)
class StageConfig:
    stage: int
    stimulus_mix: dict[str, float]
    reward_overrides: dict[str, float] = field(default_factory=dict)
    entry_condition: str = "stability_detected"
    exit_condition: str = "stability_detected"
    max_trials: Optional[int] = None
    conflict: Optional[ConflictConfig] = None
    # Fraction of trials reserved as "experimental slots". Held at the same
    # value in every condition so all of them share one calendar: slot k
    # falls on the same trial index everywhere, and only its CONTENT
    # differs. BASELINE keeps the slots and fills them with ordinary
    # REAL/FALSA, which is what makes its pi_1_pre/pi_1_post a
    # time-matched control rather than an undefined quantity.
    # None means "derive it from the special entries in stimulus_mix".
    slot_fraction: Optional[float] = None


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    run_name: str = "self_fly_run"
    reward: RewardConfig = field(default_factory=RewardConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    stages: list[StageConfig] = field(default_factory=list)
    # Phase B: which agent implementation and which experimental condition
    # produced this run. Both are plain strings (not enums) so a config.json
    # from before these fields existed still loads via config_from_dict's
    # .get(..., "baseline") fallback.
    agent_type: str = "baseline"
    experimental_condition: str = "baseline"
    control_stimulus: ControlStimulusConfig = field(default_factory=ControlStimulusConfig)
    self_variant: SelfVariantConfig = field(default_factory=SelfVariantConfig)
    connectome: ConnectomeConfig = field(default_factory=ConnectomeConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
