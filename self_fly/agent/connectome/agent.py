from __future__ import annotations

import numpy as np

from self_fly.config.schema import ConnectomeConfig
from self_fly.stimuli.features import FEATURE_ORDER

from ..actions import ACTIONS, Action
from ..common import sample_action
from .layers import ActionValuation, ForwardTrace, RecurrentIntermediate, SensoryProcessing
from .learner import ConnectomeREINFORCEUpdate

_SIGNATURE_NORM_ORDER = {"l2_column_norm": 2, "l1_column_norm": 1}


class ConnectomeInspiredAgent:
    """Reduced connectome-inspired architecture:

        inputs -> SensoryProcessing -> RecurrentIntermediate (genuine
        recurrent memory across trials) -> ActionValuation ->
        ActionSelection (softmax + sample) -> RewardModulation
        (REINFORCE, baseline-subtracted).

    Each layer's biological motivation is documented on the layer itself
    (layers.py); none of them claim fidelity to any specific circuit --
    "inspired by" means the pipeline shape, not a reproduction. The
    hidden dimension, the tanh, the leaky integrator and the truncation
    horizon of the gradient are all our choices with no biological basis.

    The sensory and output layers are trained (see learner.py); the
    recurrent weights are fixed by default. The agent's own signature is
    derived from the sensory layer, so it reflects what the agent learned
    rather than how it was initialized.

    Two distinct forward paths, and the distinction is load-bearing:
      - `probabilities()` advances the live recurrent state and records a
        ForwardTrace. Used for real trials only.
      - `evaluate()` is side-effect free. Used by every metric. A metric
        must never become part of the agent's experience.
    """

    def __init__(
        self,
        config: ConnectomeConfig,
        n_inputs: int = len(FEATURE_ORDER),
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        scheme, scale = config.init_scheme, config.initial_weight_scale
        self.config = config
        self.sensory = SensoryProcessing(n_inputs, config.hidden_dim, rng, scheme, scale)
        self.recurrent = RecurrentIntermediate(
            config.hidden_dim, config.recurrent_leak, rng, scheme, scale
        )
        self.valuation = ActionValuation(config.hidden_dim, len(ACTIONS), rng, scheme, scale)
        self._learner = ConnectomeREINFORCEUpdate(config)
        self.hidden_state = self.recurrent.zero_state()
        self._trace: ForwardTrace | None = None

    # --- live path: advances the recurrent state ---------------------------

    def probabilities(self, inputs: np.ndarray) -> np.ndarray:
        sensory_output = self.sensory.forward(inputs)
        candidate, new_hidden = self.recurrent.step(self.hidden_state, sensory_output)
        self._trace = ForwardTrace(
            inputs=np.asarray(inputs, dtype=float).copy(),
            sensory=sensory_output,
            candidate=candidate,
            hidden_prev=self.hidden_state,
            hidden=new_hidden,
        )
        self.hidden_state = new_hidden
        return self.valuation.probabilities(new_hidden)

    def sample(self, probs: np.ndarray, rng: np.random.Generator) -> Action:
        return sample_action(probs, rng)

    def update(
        self, inputs: np.ndarray, action_idx: int, reward: float, probs: np.ndarray
    ) -> None:
        assert self._trace is not None, "update() called before probabilities()"
        self._learner.update(
            self.sensory, self.recurrent, self.valuation,
            self._trace, action_idx, reward, probs,
        )

    # --- observation path: no side effects ---------------------------------

    def evaluate(
        self, inputs_batch, conditioning_state: np.ndarray | None = None
    ) -> np.ndarray:
        """Action distributions for a batch of inputs, every one of them
        evaluated from the SAME conditioning state (the live state by
        default, copied). Order-invariant by construction, and the
        agent's own state is never touched."""
        base = self.hidden_state if conditioning_state is None else conditioning_state
        base = np.asarray(base, dtype=float)
        out = []
        for inputs in inputs_batch:
            sensory_output = self.sensory.forward(np.asarray(inputs, dtype=float))
            _, hidden = self.recurrent.step(base, sensory_output)
            out.append(self.valuation.probabilities(hidden))
        return np.array(out)

    def recurrent_state(self) -> np.ndarray:
        return self.hidden_state.copy()

    # --- introspection / serialization -------------------------------------

    def weight_norm_scalar(self) -> float:
        return float(
            np.linalg.norm(self.valuation.W)
            + np.linalg.norm(self.sensory.W)
            + np.linalg.norm(self.recurrent.W_rec)
            + np.linalg.norm(self.recurrent.W_in)
        )

    def self_signature(self, transform: str = "l2_column_norm") -> np.ndarray:
        """Derived from the TRAINED sensory layer, so two agents that
        learned different policies produce different signatures."""
        return np.linalg.norm(self.sensory.W, ord=_SIGNATURE_NORM_ORDER[transform], axis=0)

    def snapshot_payload(self) -> dict:
        return {"weights": self.valuation.W.tolist(), "bias": self.valuation.b.tolist()}

    def parameters(self) -> dict:
        return {
            "sensory_W": self.sensory.W.copy(),
            "W_rec": self.recurrent.W_rec.copy(),
            "W_in": self.recurrent.W_in.copy(),
            "valuation_W": self.valuation.W.copy(),
            "valuation_b": self.valuation.b.copy(),
        }

    def set_parameters(self, params: dict) -> None:
        self.sensory.W = params["sensory_W"].copy()
        self.recurrent.W_rec = params["W_rec"].copy()
        self.recurrent.W_in = params["W_in"].copy()
        self.valuation.W = params["valuation_W"].copy()
        self.valuation.b = params["valuation_b"].copy()

    def full_state(self) -> dict:
        """Everything needed to reconstruct this agent's behaviour --
        including the recurrent state and the learner's baseline, without
        which a restored agent would not continue the same trajectory."""
        return {
            "agent_type": "connectome",
            "sensory_W": self.sensory.W.tolist(),
            "recurrent_W_rec": self.recurrent.W_rec.tolist(),
            "recurrent_W_in": self.recurrent.W_in.tolist(),
            "valuation_W": self.valuation.W.tolist(),
            "valuation_b": self.valuation.b.tolist(),
            "hidden_state": self.hidden_state.tolist(),
            "learner_baseline": self._learner.baseline.value,
            "learner_baseline_initialized": self._learner.baseline._initialized,
        }

    def load_state(self, state: dict) -> None:
        self.sensory.W = np.array(state["sensory_W"], dtype=float)
        self.recurrent.W_rec = np.array(state["recurrent_W_rec"], dtype=float)
        self.recurrent.W_in = np.array(state["recurrent_W_in"], dtype=float)
        self.valuation.W = np.array(state["valuation_W"], dtype=float)
        self.valuation.b = np.array(state["valuation_b"], dtype=float)
        self.hidden_state = np.array(state["hidden_state"], dtype=float)
        self._learner.baseline.value = state["learner_baseline"]
        self._learner.baseline._initialized = state["learner_baseline_initialized"]
        self._trace = None
