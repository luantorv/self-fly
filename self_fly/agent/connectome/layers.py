from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def init_matrix(
    rng: np.random.Generator, shape: tuple[int, int], scheme: str, fixed_scale: float
) -> np.ndarray:
    """1/sqrt(fan_in) scaling under "xavier"; a flat `fixed_scale` otherwise.
    fan_in is shape[1] (matrices are applied as W @ input)."""
    if scheme == "xavier":
        return rng.normal(0.0, 1.0 / np.sqrt(shape[1]), size=shape)
    if scheme == "fixed_scale":
        return rng.normal(0.0, fixed_scale, size=shape)
    raise ValueError(f"unknown init_scheme={scheme!r}; expected 'xavier' or 'fixed_scale'")


@dataclass
class ForwardTrace:
    """Everything the learner needs to backpropagate one step, captured
    during the forward pass so `update()` never has to recompute it (a
    second forward pass would advance the recurrent state a second time)."""

    inputs: np.ndarray
    sensory: np.ndarray
    candidate: np.ndarray
    hidden_prev: np.ndarray
    hidden: np.ndarray


class SensoryProcessing:
    """inputs -> hidden_dim, via a linear projection + tanh.

    Loosely motivated by early sensory relay circuits compressing and
    recombining raw receptor signals into a smaller working
    representation. This is a name for the role the layer plays in the
    pipeline, not a claim that it reproduces any specific identified
    circuit.

    Trainable (see ConnectomeConfig.train_sensory): the agent's own
    signature is derived from this matrix, so it has to reflect what the
    agent learned rather than how it was initialized.
    """

    def __init__(self, n_inputs: int, hidden_dim: int, rng, scheme: str, fixed_scale: float):
        self.W = init_matrix(rng, (hidden_dim, n_inputs), scheme, fixed_scale)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(self.W @ inputs)


class RecurrentIntermediate:
    """A leaky integrator carrying state between successive inputs -- the
    agent's genuine memory: its output at any trial depends on the whole
    preceding sequence, not only on the current input.

    Loosely motivated by recurrent/integrator circuits; not a claim of
    correspondence to any identified neuron population.

    `step` is PURE: it takes the previous state explicitly and returns the
    new one without storing anything. The live state lives in the agent,
    which advances it only on real trials -- this is what lets metrics
    evaluate the policy without the measurement becoming experience.
    """

    def __init__(self, hidden_dim: int, leak: float, rng, scheme: str, fixed_scale: float):
        self.hidden_dim = hidden_dim
        self.leak = leak
        self.W_rec = init_matrix(rng, (hidden_dim, hidden_dim), scheme, fixed_scale)
        self.W_in = init_matrix(rng, (hidden_dim, hidden_dim), scheme, fixed_scale)

    def step(
        self, hidden_prev: np.ndarray, sensory_output: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(candidate, new_hidden). No side effects."""
        candidate = np.tanh(self.W_rec @ hidden_prev + self.W_in @ sensory_output)
        new_hidden = (1.0 - self.leak) * hidden_prev + self.leak * candidate
        return candidate, new_hidden

    def zero_state(self) -> np.ndarray:
        return np.zeros(self.hidden_dim)


class ActionValuation:
    """hidden state -> one value per action, via a linear projection.

    Loosely motivated by convergence onto action-selective output
    neurons; not a claim of anatomical fidelity. Deliberately exposes the
    same .W/.b attribute shape as agent/baseline/policy.py's
    LinearSoftmaxPolicy.
    """

    def __init__(self, hidden_dim: int, n_actions: int, rng, scheme: str, fixed_scale: float):
        self.W = init_matrix(rng, (n_actions, hidden_dim), scheme, fixed_scale)
        self.b = np.zeros(n_actions)

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        return self.W @ hidden + self.b

    def probabilities(self, hidden: np.ndarray) -> np.ndarray:
        shifted = self.logits(hidden)
        shifted = shifted - shifted.max()
        exp = np.exp(shifted)
        return exp / exp.sum()
