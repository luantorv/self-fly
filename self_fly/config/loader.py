from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import (
    ConflictConfig,
    ConnectomeConfig,
    ControlStimulusConfig,
    ExperimentConfig,
    LearnerConfig,
    ProtocolConfig,
    RewardConfig,
    SelfVariantConfig,
    StabilityConfig,
    StageConfig,
    StimulusConfig,
)


def _stage_config_from_dict(data: dict) -> StageConfig:
    conflict_data = data.get("conflict")
    conflict = None
    if conflict_data:
        # JSON has no tuple type -- categories round-trips as a list, but
        # ConflictConfig.categories is a tuple, so dataclass equality
        # (round-trip tests) needs it cast back explicitly.
        conflict = ConflictConfig(**{**conflict_data, "categories": tuple(conflict_data["categories"])})
    return StageConfig(
        **{**{k: v for k, v in data.items() if k != "conflict"}, "conflict": conflict}
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
        stages=[_stage_config_from_dict(s) for s in data["stages"]],
        # .get() with the ExperimentConfig default, not data[...]: configs
        # saved before Phase B added these fields must still load.
        agent_type=data.get("agent_type", "baseline"),
        experimental_condition=data.get("experimental_condition", "baseline"),
        control_stimulus=ControlStimulusConfig(**data.get("control_stimulus", {})),
        self_variant=SelfVariantConfig(**data.get("self_variant", {})),
        connectome=ConnectomeConfig(**data.get("connectome", {})),
        protocol=ProtocolConfig(**data.get("protocol", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config_to_dict(config), indent=2, ensure_ascii=False))


def load_config(path: str | Path) -> ExperimentConfig:
    return config_from_dict(json.loads(Path(path).read_text()))
