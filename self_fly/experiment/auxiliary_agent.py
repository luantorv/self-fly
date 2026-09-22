from __future__ import annotations

from dataclasses import replace

from self_fly.agent.interface import Agent
from self_fly.config.schema import ExperimentConfig, StageConfig

from .stage_outcome import StageOutcome
from .stages import Stage


def train_auxiliary_agent_to_stability(
    config: ExperimentConfig,
    seed: int,
    stage0_config: StageConfig,
    max_trials: int,
) -> tuple[Agent, StageOutcome, float]:
    """Trains a second, fully isolated agent on the same REAL/FALSA task to
    stand in for "another agent" (OTHER). Reuses ExperimentEngine wholesale
    (its own RNG stream from `seed`, its own StabilityDetector, its own
    StimulusGenerator) rather than reimplementing the learning loop --
    nothing here shares state with the main experiment's agent or RNG.

    Returns the accuracy it actually reached alongside the outcome, so the
    caller can refuse a signature that does not represent a learned
    policy. If the agent does not stabilize within max_trials, the caller
    must not silently treat it as valid: this returns whatever was
    actually reached, following the same no-forcing rule as stage 1.
    """
    from .engine import ExperimentEngine  # local: engine.py imports this module too

    auxiliary_config = replace(
        config,
        seed=seed,
        run_name=f"{config.run_name}_auxiliary",
        stages=[stage0_config],
        # Never "self_multi": that would make the auxiliary engine spawn
        # its own auxiliary agent to build its own OTHER, recursing forever.
        experimental_condition="baseline",
    )
    engine = ExperimentEngine(auxiliary_config)

    trials_run = 0
    while engine.stage_machine.current_stage == Stage.STAGE_0_LEARNING and trials_run < max_trials:
        engine.step()
        trials_run += 1

    if "pi_0" in engine.policy_snapshots:
        outcome = StageOutcome.STABLE
    elif engine.stability_detector.max_consecutive_stable_observed > 0:
        outcome = StageOutcome.UNSTABLE
    else:
        outcome = StageOutcome.TIMEOUT

    return engine.agent, outcome, engine.task_accuracy()
