#!/usr/bin/env python
"""tkinter GUI for SELF-FLY: drives the same ExperimentEngine as run_experiment.py."""

from __future__ import annotations

import argparse
import dataclasses
import tkinter as tk
from pathlib import Path

from self_fly.config.defaults import default_config
from self_fly.config.loader import load_config
from self_fly.config.schema import ExperimentConfig
from self_fly.io.run_manager import RunManager
from self_fly.visualization.gui import SelfFlyApp


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config) if args.config else default_config()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    return dataclasses.replace(config, **overrides) if overrides else config


def main() -> None:
    parser = argparse.ArgumentParser(description="SELF-FLY tkinter GUI")
    parser.add_argument("--config", type=Path, default=None, help="Path to a saved ExperimentConfig JSON")
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    parser.add_argument("--run-name", type=str, default=None, help="Override the config's run_name")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Base directory for run outputs")
    parser.add_argument("--no-log", action="store_true", help="Don't persist a run directory")
    parser.add_argument("--step-delay-ms", type=int, default=30, help="Delay between engine steps")
    args = parser.parse_args()

    config = build_config(args)
    run_manager = None if args.no_log else RunManager(config, base_dir=args.runs_dir)

    root = tk.Tk()
    app = SelfFlyApp(root, config, run_manager=run_manager, step_delay_ms=args.step_delay_ms)
    try:
        root.mainloop()
    finally:
        if run_manager is not None:
            run_manager.finalize()
        del app


if __name__ == "__main__":
    main()
