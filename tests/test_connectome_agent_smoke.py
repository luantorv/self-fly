from dataclasses import replace

from self_fly.config.defaults import default_config
from self_fly.experiment.engine import ExperimentEngine
from self_fly.stimuli.features import FEATURE_ORDER


def test_connectome_agent_runs_through_the_engine_without_exceptions():
    config = replace(default_config(seed=3), agent_type="connectome")
    engine = ExperimentEngine(config)

    trials = engine.run(500)

    assert len(trials) == 500
    for trial in trials:
        assert abs(sum(trial.action_probs) - 1.0) < 1e-6
        assert all(0.0 <= p <= 1.0 for p in trial.action_probs)

    assert engine.agent.self_signature().shape == (len(FEATURE_ORDER),)
