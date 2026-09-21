#!/usr/bin/env python
"""Headless SELF-FLY runner: same ExperimentEngine the GUI drives, no tkinter import."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from self_fly.analysis.aggregate import aggregate_runs, compare_conditions
from self_fly.analysis.report import generate_report
from self_fly.config.defaults import (
    condition_control_config,
    condition_multi_self_config,
    condition_self_config,
    default_config,
)
from self_fly.config.loader import load_config
from self_fly.config.schema import ExperimentConfig
from self_fly.experiment.engine import ExperimentEngine
from self_fly.experiment.entropy import mean_policy_entropy
from self_fly.io.run_manager import RunManager
from self_fly.metrics.collectors import MetricsCollector
from self_fly.visualization.plots import generate_all_plots

CONDITIONS = ("baseline", "control", "self", "self_multi")

_CONDITION_CONFIG_BUILDERS = {
    "baseline": default_config,
    "control": condition_control_config,
    "self": condition_self_config,
    "self_multi": condition_multi_self_config,
}


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    condition = getattr(args, "condition", None)
    if args.config:
        config = load_config(args.config)
    else:
        config = _CONDITION_CONFIG_BUILDERS.get(condition, default_config)()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if getattr(args, "agent", None) is not None:
        overrides["agent_type"] = args.agent
    if condition is not None and condition != "all":
        overrides["experimental_condition"] = condition
    return dataclasses.replace(config, **overrides) if overrides else config


def parse_seeds(spec: str) -> list[int]:
    """"0:9" -> [0..9] inclusive; "0,2,5" -> [0, 2, 5]."""
    if ":" in spec:
        start, end = spec.split(":")
        return list(range(int(start), int(end) + 1))
    return [int(token) for token in spec.split(",") if token]


def run(args: argparse.Namespace) -> dict:
    config = build_config(args)
    run_manager = RunManager(config, base_dir=args.runs_dir)
    metrics = MetricsCollector()
    engine = ExperimentEngine(config, logger=run_manager.logger, metrics=metrics)

    engine.run(args.n_trials)

    pi2_name = next((name for name in engine.policy_snapshots if name.startswith("pi_2")), None)

    def _entropy_at(name: str | None) -> float | None:
        return mean_policy_entropy(engine.policy_snapshots[name]) if name in engine.policy_snapshots else None

    summary = {
        "run_id": run_manager.run_id,
        "trial_count": metrics.trial_count,
        "cumulative_reward": metrics.cumulative_reward,
        "mean_reward": metrics.mean_reward(),
        "mean_entropy": metrics.mean_entropy() if metrics.entropy_history else None,
        "entropy_at_pi0": _entropy_at("pi_0"),
        "entropy_at_pi1": _entropy_at("pi_1"),
        "entropy_at_pi2": _entropy_at(pi2_name),
        "action_frequencies": metrics.action_frequencies(),
        "stage_reached": engine.current_stage,
        "experiment_finished": engine.stage_machine.finished,
        "policy_snapshots_captured": list(engine.policy_snapshots.keys()),
    }
    run_manager.write_metrics_summary(summary)
    generate_all_plots(run_manager.logger.trials_path, run_manager.run_dir / "plots")
    run_manager.finalize()
    # After finalize(): generate_report() reads manifest.json's status/
    # finished_at, only populated once finalize() has run.
    (run_manager.run_dir / "report.md").write_text(
        generate_report(run_manager.run_dir), encoding="utf-8"
    )
    summary["run_dir"] = str(run_manager.run_dir)
    return summary


def _run_seeds(args: argparse.Namespace, seeds: list[int]) -> list[Path]:
    run_dirs = []
    for seed in seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        summary = run(seed_args)
        run_dirs.append(Path(summary["run_dir"]))
    return run_dirs


def run_multi_seed(args: argparse.Namespace) -> dict:
    """Runs the same config once per seed (reusing run() unchanged) and
    returns a purely descriptive aggregate -- a seed that fails to
    stabilize still contributes its run, it is never dropped or retried."""
    return aggregate_runs(_run_seeds(args, parse_seeds(args.seeds)))


def run_all_conditions(args: argparse.Namespace) -> dict:
    """--condition all: runs every named condition across the same seeds
    and returns a purely descriptive per-condition comparison (see
    compare_conditions) -- never a ranking or a "winner"."""
    seeds = parse_seeds(args.seeds)
    runs_by_condition = {}
    for condition in CONDITIONS:
        condition_args = argparse.Namespace(**vars(args))
        condition_args.condition = condition
        runs_by_condition[condition] = _run_seeds(condition_args, seeds)
    return compare_conditions(runs_by_condition)


def main() -> None:
    parser = argparse.ArgumentParser(description="SELF-FLY headless experiment runner")
    parser.add_argument("--config", type=Path, default=None, help="Path to a saved ExperimentConfig JSON")
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    parser.add_argument("--seeds", type=str, default=None, help='Run every seed in "0:9" or "0,1,2" and aggregate')
    parser.add_argument("--agent", type=str, default=None, choices=["baseline", "connectome"], help="Agent implementation")
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=[*CONDITIONS, "all"],
        help="Experimental condition; 'all' runs every condition and compares them (needs --seeds)",
    )
    parser.add_argument("--n-trials", type=int, default=2000, help="Number of trials to run")
    parser.add_argument("--run-name", type=str, default=None, help="Override the config's run_name")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Base directory for run outputs")
    args = parser.parse_args()

    if args.condition == "all":
        if not args.seeds:
            parser.error("--condition all requires --seeds")
        summary = run_all_conditions(args)
    elif args.seeds:
        summary = run_multi_seed(args)
    else:
        summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
