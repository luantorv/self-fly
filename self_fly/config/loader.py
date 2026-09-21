from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import (
    ExperimentConfig,
    LearnerConfig,
    RewardConfig,
    StabilityConfig,
    StageConfig,
    StimulusConfig,
)


def config_to_dict(config: ExperimentConfig) -> dict:
    return asdict(config)


def config_from_dict(data: dict) -> ExperimentConfig:
    return ExperimentConfig(
        seed=data["seed"],
        run_name=data["run_name"],
        reward=RewardConfig(**data["reward"]),
        stability=StabilityConfig(**data["stability"]),
        stimulus=StimulusConfig(**data["stimulus"]),
        learner=LearnerConfig(**data["learner"]),
        stages=[StageConfig(**s) for s in data["stages"]],
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False))


def load_config(path: str | Path) -> ExperimentConfig:
    return config_from_dict(json.loads(Path(path).read_text()))
