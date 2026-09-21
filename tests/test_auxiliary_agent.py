from dataclasses import replace

from self_fly.config.defaults import default_config
from self_fly.config.schema import StabilityConfig
from self_fly.experiment.auxiliary_agent import train_auxiliary_agent_to_stability
from self_fly.experiment.stage_outcome import StageOutcome


def test_auxiliary_agent_reaches_stable_with_a_generous_budget():
    fast_stability = StabilityConfig(
        window_size=15,
        tv_threshold=0.35,
        reward_delta_threshold=0.6,
        cv_threshold=1.5,
        consistency_threshold=0.1,
        consecutive_windows_required=2,
    )
    config = replace(default_config(seed=0), stability=fast_stability)
    stage0 = config.stages[0]

    agent, outcome = train_auxiliary_agent_to_stability(
        config, seed=999, stage0_config=stage0, max_trials=5000
    )

    assert outcome == StageOutcome.STABLE
    assert agent.policy.W.shape == (3, 10)


def test_auxiliary_agent_never_silently_accepted_with_a_tiny_budget():
    config = default_config(seed=0)
    stage0 = config.stages[0]

    agent, outcome = train_auxiliary_agent_to_stability(
        config, seed=999, stage0_config=stage0, max_trials=3
    )

    # Never crashes, never pretends to have converged -- an honest, checkable outcome instead.
    assert outcome in (StageOutcome.TIMEOUT, StageOutcome.UNSTABLE)
