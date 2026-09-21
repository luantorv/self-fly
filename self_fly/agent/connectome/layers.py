from __future__ import annotations

import numpy as np


class SensoryProcessing:
    """features -> hidden_dim, via a fixed random linear projection + tanh.

    Loosely motivated by early sensory relay circuits compressing and
    recombining raw receptor signals into a smaller working
    representation. This is a name for the role the layer plays in the
    pipeline, not a claim that it reproduces any specific identified
    circuit -- see ConnectomeInspiredAgent's docstring for why these
    weights are fixed rather than trained.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        rng: np.random.Generator,
        initial_weight_scale: float,
    ):
        self.W = rng.normal(0.0, initial_weight_scale, size=(hidden_dim, n_features))

    def forward(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(self.W @ features)


class RecurrentIntermediate:
    """A leaky integrator: hidden_state persists across trials, updated by
    both the current sensory signal and its own previous value. This is
    the agent's genuine memory -- its output at trial t depends on every
    trial before it, not only on the current stimulus.

    Loosely motivated by recurrent/integrator circuits that carry state
    between successive inputs; not a claim of correspondence to any
    specific identified neuron population.
    """

    def __init__(
        self,
        hidden_dim: int,
        leak: float,
        rng: np.random.Generator,
        initial_weight_scale: float,
    ):
        self.hidden_dim = hidden_dim
        self.leak = leak
        self.W_rec = rng.normal(0.0, initial_weight_scale, size=(hidden_dim, hidden_dim))
        self.W_in = rng.normal(0.0, initial_weight_scale, size=(hidden_dim, hidden_dim))
        self.hidden_state = np.zeros(hidden_dim)

    def forward(self, sensory_output: np.ndarray) -> np.ndarray:
        candidate = np.tanh(self.W_rec @ self.hidden_state + self.W_in @ sensory_output)
        self.hidden_state = (1 - self.leak) * self.hidden_state + self.leak * candidate
        return self.hidden_state

    def reset(self) -> None:
        self.hidden_state = np.zeros(self.hidden_dim)


class ActionValuation:
    """hidden_state -> one value per action, via a linear projection --
    the only trained layer (see ConnectomeInspiredAgent). Loosely
    motivated by convergence onto action-selective output neurons; not a
    claim of anatomical fidelity. Deliberately exposes the same .W/.b
    attribute shape as agent/baseline/policy.py's LinearSoftmaxPolicy so
    agent/baseline/learner.py's REINFORCEUpdate can be reused unmodified.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_actions: int,
        rng: np.random.Generator,
        initial_weight_scale: float,
    ):
        self.W = rng.normal(0.0, initial_weight_scale, size=(n_actions, hidden_dim))
        self.b = np.zeros(n_actions)

    def logits(self, hidden_state: np.ndarray) -> np.ndarray:
        return self.W @ hidden_state + self.b

    def probabilities(self, hidden_state: np.ndarray) -> np.ndarray:
        shifted_logits = self.logits(hidden_state)
        shifted_logits = shifted_logits - shifted_logits.max()
        exp = np.exp(shifted_logits)
        return exp / exp.sum()
