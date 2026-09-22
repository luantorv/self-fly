from __future__ import annotations

import numpy as np

from self_fly.config.schema import ConnectomeConfig

from ..baseline.learner import RunningMeanBaseline
from .layers import ActionValuation, ForwardTrace, RecurrentIntermediate, SensoryProcessing


class ConnectomeREINFORCEUpdate:
    """Advantage-weighted policy gradient propagated through all three
    layers, instead of only through the output layer.

    This exists separately from agent/baseline/learner.py's single-layer
    REINFORCEUpdate (whose behaviour must stay bit-identical) because the
    sensory layer has to be trainable: the agent's own signature is
    derived from it, so with a frozen sensory matrix SELF_A would encode
    the random initialization rather than anything the agent learned.

    Gradient, with g = onehot(a) - probs and a stop-gradient on the
    previous hidden state (truncation horizon 1 -- a standard, biased but
    cheap approximation; full BPTT would require retaining the whole
    trial history):

        dW_v   = outer(g, h)
        delta_h   = W_v^T g
        delta_c   = leak * delta_h * (1 - candidate^2)
        dW_in  = outer(delta_c, sensory)      dW_rec = outer(delta_c, hidden_prev)
        delta_s   = (W_in^T delta_c) * (1 - sensory^2)
        dW_s   = outer(delta_s, inputs)
    """

    def __init__(self, config: ConnectomeConfig):
        self.learning_rate = config.learning_rate
        self.leak = config.recurrent_leak
        self.train_sensory = config.train_sensory
        self.train_recurrent = config.train_recurrent
        self.baseline = RunningMeanBaseline(config.baseline_momentum)

    def update(
        self,
        sensory: SensoryProcessing,
        recurrent: RecurrentIntermediate,
        valuation: ActionValuation,
        trace: ForwardTrace,
        action_idx: int,
        reward: float,
        probs: np.ndarray,
    ) -> None:
        advantage = reward - self.baseline.value
        step = self.learning_rate * advantage

        grad_logits = -probs.copy()
        grad_logits[action_idx] += 1.0

        valuation.W += step * np.outer(grad_logits, trace.hidden)
        valuation.b += step * grad_logits

        if self.train_sensory or self.train_recurrent:
            delta_hidden = valuation.W.T @ grad_logits
            delta_candidate = self.leak * delta_hidden * (1.0 - trace.candidate**2)

            if self.train_recurrent:
                recurrent.W_in += step * np.outer(delta_candidate, trace.sensory)
                recurrent.W_rec += step * np.outer(delta_candidate, trace.hidden_prev)

            if self.train_sensory:
                delta_sensory = (recurrent.W_in.T @ delta_candidate) * (1.0 - trace.sensory**2)
                sensory.W += step * np.outer(delta_sensory, trace.inputs)

        self.baseline.update(reward)
