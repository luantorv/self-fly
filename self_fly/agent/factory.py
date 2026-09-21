from __future__ import annotations

from self_fly.config.schema import ConnectomeConfig, LearnerConfig
from self_fly.stimuli.features import FEATURE_ORDER

from .baseline.agent import BaselineAgent
from .connectome.agent import ConnectomeInspiredAgent
from .interface import Agent


def make_agent(
    agent_type: str,
    learner_config: LearnerConfig,
    connectome_config: ConnectomeConfig,
    seed: int = 0,
    n_features: int = len(FEATURE_ORDER),
) -> Agent:
    if agent_type == "baseline":
        return BaselineAgent(learner_config, n_features)
    if agent_type == "connectome":
        return ConnectomeInspiredAgent(connectome_config, n_features, seed)
    raise ValueError(f"unknown agent_type={agent_type!r}; expected 'baseline' or 'connectome'")
