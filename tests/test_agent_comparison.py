import argparse
from dataclasses import replace

import run_experiment
from self_fly.analysis.aggregate import compare_conditions
from self_fly.config.defaults import default_config
from self_fly.config.loader import save_config
from stability_helpers import fast_stability_config


def _fast(config, stage1_max_trials: int = 100):
    fast_stability = fast_stability_config()
    stage0, stage1 = config.stages
    stage1 = replace(stage1, max_trials=stage1_max_trials)
    return replace(config, stability=fast_stability, stages=[stage0, stage1])


def _run_dirs_for(agent_type, seeds, tmp_path):
    run_dirs = []
    for seed in seeds:
        config = replace(_fast(default_config(seed=seed)), agent_type=agent_type)
        config_path = tmp_path / f"{agent_type}_{seed}.json"
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


def test_compare_conditions_groups_by_agent_type_without_ranking(tmp_path):
    """compare_conditions() is generic over whatever its dict keys mean --
    block 9 grouped by experimental_condition, this groups the exact same
    function by agent_type, no new code needed."""
    baseline_dirs = _run_dirs_for("baseline", [0, 1], tmp_path)
    connectome_dirs = _run_dirs_for("connectome", [0, 1], tmp_path)

    result = compare_conditions({"baseline": baseline_dirs, "connectome": connectome_dirs})

    assert set(result.keys()) == {"baseline", "connectome"}
    for agent_result in result.values():
        assert agent_result["n_seeds"] == 2
        assert "aggregated" in agent_result

    forbidden = ("rank", "best", "winner")
    assert not any(any(term in str(k).lower() for term in forbidden) for k in result)
