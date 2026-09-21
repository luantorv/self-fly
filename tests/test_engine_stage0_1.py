from self_fly.config.defaults import default_config
from self_fly.experiment.engine import ExperimentEngine
from self_fly.metrics.collectors import MetricsCollector


def test_engine_runs_200_headless_steps_without_exceptions():
    config = default_config(seed=0)
    metrics = MetricsCollector()
    engine = ExperimentEngine(config, metrics=metrics)

    trials = engine.run(200)

    assert len(trials) == 200
    assert metrics.trial_count == 200
    assert all(t.stage == 0 for t in trials)
    assert all(t.action in ("REAL", "FALSA", "NO_SE") for t in trials)
    assert all(t.trial_index == i for i, t in enumerate(trials))


def test_engine_trial_schema_fields_present():
    config = default_config(seed=1)
    engine = ExperimentEngine(config)
    trial = engine.step()

    assert isinstance(trial.features, tuple) and len(trial.features) == 10
    assert len(trial.action_probs) == 3
    assert abs(sum(trial.action_probs) - 1.0) < 1e-9
    assert trial.ground_truth_label in ("real", "falsa")
