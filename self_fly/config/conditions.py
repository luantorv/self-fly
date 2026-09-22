"""The five experimental conditions, built from one shared skeleton.

Each condition differs from the previous one in exactly ONE property, so
that a difference between adjacent conditions is attributable:

    BASELINE   slots filled with ordinary rewarded REAL/FALSA
    FRESH      + the slot carries no reward       (adds: zero reward)
    CONTROL    + the slot stimulus is frozen      (adds: repetition)
    OTHER      + derived from another agent's weights
                                                  (adds: generative structure)
    SELF       + derived from THIS agent's weights
                                                  (adds: authorship)

Every condition keeps 80% ordinary REAL/FALSA and a 20% slot, and every
condition shares the same slot calendar, so none of them gets more or less
ordinary training than another and slot k falls on the same trial index
everywhere. Without FRESH the step from BASELINE to CONTROL would change
two properties at once and nothing in between could be attributed.
"""

from __future__ import annotations

from dataclasses import replace

from .schema import ExperimentConfig, StageConfig

SLOT_FRACTION = 0.20
_ORDINARY = {"real": 0.40, "falsa": 0.40}

# Every special category is worth exactly zero whatever the agent does:
# the slot is an observation, not a task.
_ZERO_REWARD_ACTIONS = ("REAL", "FALSA", "NO_SE")


def _zero_rewards(*categories: str) -> dict[str, float]:
    return {
        f"{category}|{action}": 0.0
        for category in categories
        for action in _ZERO_REWARD_ACTIONS
    }


def _stage_0() -> StageConfig:
    return StageConfig(
        stage=0,
        stimulus_mix={"real": 0.5, "falsa": 0.5},
        reward_overrides={},
        entry_condition="initial",
        exit_condition="stability_detected",
        slot_fraction=0.0,
    )


def _stage_1(specials: dict[str, float], max_trials: int = 5000) -> StageConfig:
    return StageConfig(
        stage=1,
        stimulus_mix={**_ORDINARY, **specials},
        reward_overrides=_zero_rewards(*specials),
        entry_condition="stability_detected",
        exit_condition="stability_detected_or_max_trials",
        max_trials=max_trials,
        slot_fraction=SLOT_FRACTION,
    )


def _condition(
    name: str, specials: dict[str, float], seed: int, run_name: str | None
) -> ExperimentConfig:
    return ExperimentConfig(
        seed=seed,
        run_name=run_name or f"self_fly_{name}",
        stages=[_stage_0(), _stage_1(specials)],
        agent_type="baseline",
        experimental_condition=name,
    )


def baseline_condition(seed: int = 0, run_name: str | None = None) -> ExperimentConfig:
    """No special stimulus at all. The slots still exist and still fall on
    the same trial indices as everywhere else -- they are simply filled
    with ordinary rewarded REAL/FALSA. That is what gives BASELINE a
    pi_1_pre/pi_1_post pair at a matched moment, so the other conditions
    have a time-matched control instead of an incomparable one."""
    return _condition("baseline", {}, seed, run_name)


def fresh_condition(seed: int = 0, run_name: str | None = None) -> ExperimentConfig:
    return _condition("fresh", {"fresh": SLOT_FRACTION}, seed, run_name)


def control_condition(seed: int = 0, run_name: str | None = None) -> ExperimentConfig:
    return _condition("control", {"control": SLOT_FRACTION}, seed, run_name)


def other_condition(seed: int = 0, run_name: str | None = None) -> ExperimentConfig:
    config = _condition("other", {"other": SLOT_FRACTION}, seed, run_name)
    # Needs the auxiliary agents trained, same as self_multi does.
    return replace(config, experimental_condition="other")


def self_condition(seed: int = 0, run_name: str | None = None) -> ExperimentConfig:
    return _condition("self", {"self_a": SLOT_FRACTION}, seed, run_name)


def self_multi_condition(seed: int = 0, run_name: str | None = None) -> ExperimentConfig:
    """SELF_A, SELF(eps*) and OTHER share the SAME 20% slot, one third
    each. The total special fraction is unchanged, so this condition gets
    exactly as much ordinary training as the others -- the earlier 40%
    version confounded 'more special categories' with 'less REAL/FALSA'."""
    third = SLOT_FRACTION / 3
    return _condition(
        "self_multi",
        {"self_a": third, "self_b": third, "other": third},
        seed,
        run_name,
    )


CONDITION_BUILDERS = {
    "baseline": baseline_condition,
    "fresh": fresh_condition,
    "control": control_condition,
    "other": other_condition,
    "self": self_condition,
    "self_multi": self_multi_condition,
}

# Conditions that need the auxiliary agents trained, either because they
# present an OTHER-derived stimulus or because the analysis evaluates one.
CONDITIONS_REQUIRING_AUXILIARY = ("other", "self_multi")
