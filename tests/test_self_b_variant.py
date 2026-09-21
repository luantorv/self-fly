import numpy as np

from self_fly.agent.baseline.agent import BaselineAgent
from self_fly.config.schema import LearnerConfig
from self_fly.environment.environment import Environment
from self_fly.environment.stimulus_types import StimulusLabel
from self_fly.stimuli.self_stimulus import SelfStimulusGenerator


def _trained_like_agent(seed: int = 0) -> BaselineAgent:
    agent = BaselineAgent(LearnerConfig())
    rng = np.random.default_rng(seed)
    agent.policy.W = rng.normal(size=agent.policy.W.shape)
    return agent


def test_self_a_and_self_b_transforms_produce_different_features():
    agent = _trained_like_agent()
    generator = SelfStimulusGenerator()

    self_a = generator.generate(
        agent.self_signature(transform="l2_column_norm"), stage=1, label=StimulusLabel.SELF_A
    )
    self_b = generator.generate(
        agent.self_signature(transform="l1_column_norm"), stage=1, label=StimulusLabel.SELF_B
    )

    assert self_a.features.to_tuple() != self_b.features.to_tuple()
    assert self_a.label == StimulusLabel.SELF_A
    assert self_b.label == StimulusLabel.SELF_B
    assert self_a.category_id == "self_a"
    assert self_b.category_id == "self_b"
    assert self_a.stimulus_id == "self_a_frozen"
    assert self_b.stimulus_id == "self_b_frozen"


def test_self_b_and_other_labels_never_carry_through_observe():
    agent = _trained_like_agent(seed=1)
    generator = SelfStimulusGenerator()

    for label, transform in [
        (StimulusLabel.SELF_B, "l1_column_norm"),
        (StimulusLabel.OTHER, "l2_column_norm"),
    ]:
        stimulus = generator.generate(agent.self_signature(transform=transform), stage=1, label=label)
        observable = Environment.observe(stimulus)
        assert not hasattr(observable, "label")
        assert not hasattr(observable, "category_id")
        assert not hasattr(observable, "stage")
