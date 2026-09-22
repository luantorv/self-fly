from dataclasses import replace

from self_fly.config.defaults import default_config
from self_fly.experiment.auxiliary_agent import train_auxiliary_agent_to_stability
from self_fly.experiment.stage_outcome import StageOutcome
from stability_helpers import fast_stability_config


def test_auxiliary_agent_reaches_stable_with_a_generous_budget():
    config = replace(default_config(seed=0), stability=fast_stability_config())
    stage0 = config.stages[0]

    agent, outcome, accuracy = train_auxiliary_agent_to_stability(
        config, seed=999, stage0_config=stage0, max_trials=5000
    )

    assert outcome == StageOutcome.STABLE
    assert agent.policy.W.shape == (3, 10)
    assert 0.0 <= accuracy <= 1.0


def test_auxiliary_agent_never_silently_accepted_with_a_tiny_budget():
    config = default_config(seed=0)
    stage0 = config.stages[0]

    agent, outcome, accuracy = train_auxiliary_agent_to_stability(
        config, seed=999, stage0_config=stage0, max_trials=3
    )

    # Never crashes, never pretends to have converged -- an honest,
    # checkable outcome instead.
    assert outcome in (StageOutcome.TIMEOUT, StageOutcome.UNSTABLE)
    # And with 3 trials it cannot have learned the task, which is what the
    # caller checks before it will build OTHER out of this agent.
    assert accuracy < 0.70


def test_stable_under_the_real_criterion_implies_a_competent_agent():
    """Under the real stability criterion, accuracy is one of the two
    conjunctive components -- so an auxiliary agent that reports STABLE
    cannot be an untrained one. This is what lets the caller trust a
    STABLE signature as 'another agent's policy'.

    (Deliberately NOT using fast_stability_config here: that one waives
    the accuracy component, so it would stop training almost immediately
    and the property under test would not hold.)"""
    config = default_config(seed=0)
    stage0 = config.stages[0]

    _, outcome, accuracy = train_auxiliary_agent_to_stability(
        config, seed=1234, stage0_config=stage0, max_trials=20000
    )

    if outcome == StageOutcome.STABLE:
        assert accuracy >= config.stability.min_accuracy
    else:
        # Not converging is a valid result; it just must not be passed off
        # as a usable OTHER signature.
        assert accuracy < 1.0
