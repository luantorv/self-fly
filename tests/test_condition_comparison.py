import argparse
from dataclasses import replace

import run_experiment
from self_fly.analysis.aggregate import compare_conditions
from self_fly.config.defaults import condition_control_config, default_config
from self_fly.config.loader import save_config
from self_fly.config.schema import StabilityConfig


def _fast(config, stage1_max_trials: int = 100):
    fast_stability = StabilityConfig(
        window_size=15,
        tv_threshold=0.35,
        reward_delta_threshold=0.6,
        cv_threshold=1.5,
        consistency_threshold=0.1,
        consecutive_windows_required=2,
    )
    stage0, stage1 = config.stages
    stage1 = replace(stage1, max_trials=stage1_max_trials)
    return replace(config, stability=fast_stability, stages=[stage0, stage1])


def _run_dirs_for(config_builder, seeds, tmp_path, label):
    run_dirs = []
    for seed in seeds:
        config = _fast(config_builder(seed=seed))
        config_path = tmp_path / f"{label}_{seed}.json"
        save_config(config, config_path)
        args = argparse.Namespace(
            config=config_path,
            seed=None,
            seeds=None,
            agent=None,
            condition=None,
            n_trials=800,
            run_name=None,
            runs_dir=tmp_path / "runs",
        )
        summary = run_experiment.run(args)
        run_dirs.append(summary["run_dir"])
    return run_dirs


def test_compare_conditions_reports_per_condition_aggregates_without_ranking(tmp_path):
    baseline_dirs = _run_dirs_for(default_config, [0, 1], tmp_path, "baseline")
    control_dirs = _run_dirs_for(condition_control_config, [0, 1], tmp_path, "control")

    result = compare_conditions({"baseline": baseline_dirs, "control": control_dirs})

    assert set(result.keys()) == {"baseline", "control"}
    for condition_result in result.values():
        assert condition_result["n_seeds"] == 2
        assert "aggregated" in condition_result
        assert "per_seed" in condition_result

    forbidden = ("rank", "best", "winner")
    assert not any(any(term in str(k).lower() for term in forbidden) for k in result)
    for condition_result in result.values():
        assert not any(any(term in str(k).lower() for term in forbidden) for k in condition_result)
