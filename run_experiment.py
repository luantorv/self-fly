#!/usr/bin/env python
"""Headless SELF-FLY runner: same ExperimentEngine the GUI drives, no tkinter import."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from self_fly.config.defaults import default_config
from self_fly.config.loader import load_config
from self_fly.config.schema import ExperimentConfig
from self_fly.experiment.engine import ExperimentEngine
from self_fly.io.run_manager import RunManager
from self_fly.metrics.collectors import MetricsCollector
from self_fly.visualization.plots import generate_all_plots


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config) if args.config else default_config()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    return dataclasses.replace(config, **overrides) if overrides else config


def run(args: argparse.Namespace) -> dict:
    config = build_config(args)
    run_manager = RunManager(config, base_dir=args.runs_dir)
    metrics = MetricsCollector()
    engine = ExperimentEngine(config, logger=run_manager.logger, metrics=metrics)

    engine.run(args.n_trials)

    summary = {
        "run_id": run_manager.run_id,
        "trial_count": metrics.trial_count,
        "cumulative_reward": metrics.cumulative_reward,
        "mean_reward": metrics.mean_reward(),
        "action_frequencies": metrics.action_frequencies(),
        "stage_reached": engine.current_stage,
        "experiment_finished": engine.stage_machine.finished,
        "policy_snapshots_captured": list(engine.policy_snapshots.keys()),
    }
    run_manager.write_metrics_summary(summary)
    generate_all_plots(run_manager.logger.trials_path, run_manager.run_dir / "plots")
    run_manager.finalize()
    summary["run_dir"] = str(run_manager.run_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SELF-FLY headless experiment runner")
    parser.add_argument("--config", type=Path, default=None, help="Path to a saved ExperimentConfig JSON")
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    parser.add_argument("--n-trials", type=int, default=2000, help="Number of trials to run")
    parser.add_argument("--run-name", type=str, default=None, help="Override the config's run_name")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Base directory for run outputs")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
