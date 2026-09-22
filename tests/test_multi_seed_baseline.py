import argparse
from dataclasses import replace

import run_experiment
from self_fly.config.defaults import default_config
from self_fly.config.loader import save_config
from stability_helpers import fast_stability_config


def _fast_config(seed: int = 0):
    config = default_config(seed=seed)
    fast_stability = fast_stability_config()
    stage0, stage1 = config.stages
    stage1 = replace(stage1, max_trials=400)
    return replace(config, stability=fast_stability, stages=[stage0, stage1])


def test_multi_seed_baseline_aggregates_across_seeds(tmp_path):
    config_path = tmp_path / "config.json"
    save_config(_fast_config(), config_path)

    args = argparse.Namespace(
        config=config_path,
        seed=None,
        seeds="0,1,2",
        agent=None,
        condition=None,
        n_trials=800,
        run_name=None,
        runs_dir=tmp_path / "runs",
    )
    aggregate = run_experiment.run_multi_seed(args)

    assert aggregate["n_seeds"] == 3
    assert len(aggregate["per_seed"]) == 3
    assert {s["seed"] for s in aggregate["per_seed"]} == {0, 1, 2}
    assert aggregate["stage0_convergence_rate"] is not None
    assert aggregate["pi2_reached_rate"] is not None
    for field_name in ("mean_reward", "pi0_trial", "pi1_trial", "pi2_trial"):
        assert field_name in aggregate["aggregated"]


def test_aggregate_tolerates_a_seed_that_never_reaches_pi2(tmp_path):
    """A stage-1 that always times out (max_trials smaller than reachable)
    must still show up in the aggregate, not be dropped or crash it."""
    config = _fast_config()
    stage0, stage1 = config.stages
    stage1 = replace(stage1, max_trials=1)
    config = replace(config, stages=[stage0, stage1])
    config_path = tmp_path / "config.json"
    save_config(config, config_path)

    args = argparse.Namespace(
        config=config_path,
        seed=None,
        seeds="0",
        agent=None,
        condition=None,
        n_trials=300,
        run_name=None,
        runs_dir=tmp_path / "runs",
    )
    aggregate = run_experiment.run_multi_seed(args)

    assert aggregate["n_seeds"] == 1
    seed_summary = aggregate["per_seed"][0]
    assert seed_summary["pi2_trial"] is None or seed_summary["pi2_reached"] is True
