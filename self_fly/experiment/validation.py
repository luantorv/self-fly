from __future__ import annotations

import numpy as np

from self_fly.agent.actions import ACTIONS
from self_fly.config.schema import ProtocolConfig, StimulusConfig
from self_fly.environment.stimulus_types import StimulusLabel
from self_fly.stimuli.generator import StimulusGenerator

# Index of the action that is correct for each REAL/FALSA stimulus. NO_SE is
# never correct: it is a valid action but not a correct classification, so it
# counts as an error (see task_accuracy).
_CORRECT_ACTION_INDEX = {
    StimulusLabel.REAL: [a.value for a in ACTIONS].index("REAL"),
    StimulusLabel.FALSA: [a.value for a in ACTIONS].index("FALSA"),
}


class ValidationSet:
    """A fixed REAL/FALSA set, held out from the experiment: it is never
    presented to the agent as a trial and never produces reward or an
    update. Built once per protocol (not per experimental seed) so that
    accuracies from different seeds are measured on the same ruler."""

    def __init__(self, inputs: np.ndarray, correct_action: np.ndarray):
        self.inputs = inputs
        self.correct_action = correct_action

    def __len__(self) -> int:
        return len(self.correct_action)


def build_validation_set(
    stimulus_config: StimulusConfig, protocol: ProtocolConfig
) -> ValidationSet:
    # Own stream, offset from the protocol seed, so it can never collide
    # with S_ref (which uses the protocol seed directly).
    rng = np.random.default_rng(protocol.protocol_seed + 1)
    generator = StimulusGenerator(stimulus_config, rng)
    samples = [generator.sample(stage=0) for _ in range(protocol.validation_set_size)]
    return ValidationSet(
        inputs=np.array([s.features.to_tuple() for s in samples]),
        correct_action=np.array([_CORRECT_ACTION_INDEX[s.label] for s in samples]),
    )


def task_accuracy(agent, validation_set: ValidationSet) -> float:
    """Fraction of the validation set where the agent's most probable
    action is the correct classification.

    Deterministic readout (argmax), not a sampled action: this measures
    the policy's competence rather than its stochasticity, which matters
    because accuracy is used as a stability criterion and a sampling-based
    figure would inject exactly the noise that criterion is meant to
    exclude. Evaluated through `evaluate()`, so it has no side effects on
    a recurrent agent.
    """
    probs = agent.evaluate(
        validation_set.inputs, conditioning_state=agent.recurrent_state()
    )
    return float((probs.argmax(axis=1) == validation_set.correct_action).mean())
