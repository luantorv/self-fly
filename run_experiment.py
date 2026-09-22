#!/usr/bin/env python
"""Headless SELF-FLY runner: same ExperimentEngine the GUI drives, no tkinter import."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from self_fly.analysis.aggregate import aggregate_runs, compare_conditions
from self_fly.analysis.report import generate_report
from self_fly.analysis.temporal_dynamics import classify_dt_series
from self_fly.config.conditions import CONDITION_BUILDERS
from self_fly.config.defaults import default_config
from self_fly.config.loader import load_config
from self_fly.config.schema import ExperimentConfig
from self_fly.experiment.engine import ExperimentEngine
from self_fly.experiment.entropy import mean_policy_entropy
from self_fly.experiment.policy_distance import policy_distance_from_snapshots
from self_fly.experiment.special_evaluation import (
    evaluate_snapshots_on_special_stimuli,
    stimulus_space_distances,
)
from self_fly.io.run_manager import RunManager
from self_fly.metrics.collectors import MetricsCollector
from self_fly.visualization.plots import generate_all_plots

# The causal ladder of protocol §29: each step adds exactly one property
# (zero reward -> repetition -> generative structure -> authorship).
CONDITIONS = ("baseline", "fresh", "control", "other", "self")
ALL_CONDITIONS = tuple(CONDITION_BUILDERS)

_CONDITION_CONFIG_BUILDERS = CONDITION_BUILDERS


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

    trials = engine.run(args.n_trials)

    snapshots = engine.policy_snapshots
    pi2_name = next((name for name in snapshots if name.startswith("pi_2")), None)

    def _entropy_at(name: str | None) -> float | None:
        return mean_policy_entropy(snapshots[name]) if name in snapshots else None

    def _distance(a: str | None, b: str | None) -> dict | None:
        if a in snapshots and b in snapshots:
            return policy_distance_from_snapshots(snapshots[a], snapshots[b])
        return None

    special = engine.special_stimuli_features()
    deltas = [t.policy_js_delta for t in trials]

    summary = {
        "run_id": run_manager.run_id,
        "trial_count": metrics.trial_count,
        "cumulative_reward": metrics.cumulative_reward,
        "mean_reward": metrics.mean_reward(),
        "task_accuracy": engine.task_accuracy(),
        "mean_entropy": metrics.mean_entropy() if metrics.entropy_history else None,
        "entropy_at_pi0": _entropy_at("pi_0"),
        "entropy_at_pi1_pre": _entropy_at("pi_1_pre"),
        "entropy_at_pi1": _entropy_at("pi_1_post"),
        "entropy_at_pi2": _entropy_at(pi2_name),
        # Snapshot-to-snapshot distances. Previously none of these were
        # persisted, so even JS(pi_0, pi_1) required an external script.
        "policy_distances": {
            "pi0_to_pi1_pre": _distance("pi_0", "pi_1_pre"),
            "pi1_pre_to_pi1_post": _distance("pi_1_pre", "pi_1_post"),
            "pi0_to_pi1_post": _distance("pi_0", "pi_1_post"),
            "pi0_to_pi2": _distance("pi_0", pi2_name),
            "pi1_post_to_pi2": _distance("pi_1_post", pi2_name),
        },
        # The experiment's primary measurement (see special_evaluation.py).
        "special_stimulus_response": evaluate_snapshots_on_special_stimuli(
            config, snapshots, special
        ),
        "special_stimulus_distances": stimulus_space_distances(special),
        "dt_series": classify_dt_series(deltas),
        "action_frequencies": metrics.action_frequencies(),
        "action_counts_by_category": {
            category: dict(counts)
            for category, counts in metrics.action_counts_by_label.items()
        },
        "n_by_category": {
            category: sum(counts.values())
            for category, counts in metrics.action_counts_by_label.items()
        },
        "stage_reached": engine.current_stage,
        "experiment_finished": engine.stage_machine.finished,
        "policy_snapshots_captured": list(snapshots.keys()),
        "auxiliary_agent_outcome": (
            engine.other_agent_outcome.value if engine.other_agent_outcome else None
        ),
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
        choices=[*ALL_CONDITIONS, "all"],
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
