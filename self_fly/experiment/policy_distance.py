from __future__ import annotations

import numpy as np

from self_fly.config.schema import StimulusConfig
from self_fly.stimuli.generator import StimulusGenerator

from .types import PolicySnapshot


def _softmax_probs(weights: np.ndarray, bias: np.ndarray, features: np.ndarray) -> np.ndarray:
    logits = weights @ features + bias
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def tv_distance_arrays(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (sqrt of JS divergence, base-2), bounded in [0, 1]."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    js_divergence = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(js_divergence, 0.0)))


def build_reference_set(
    stimulus_config: StimulusConfig, rng: np.random.Generator
) -> list[np.ndarray]:
    """Fixed set of feature vectors used to compare policies consistently:
    S_ref, generated once per run and reused for every snapshot comparison."""
    generator = StimulusGenerator(stimulus_config, rng)
    return [
        np.array(generator.sample(stage=0).features.to_tuple())
        for _ in range(stimulus_config.reference_set_size)
    ]


class TemporalDistanceTracker:
    """Continuous D_t = JS(pi_t, pi_{t-1}) over S_ref, every trial rather
    than only at snapshot boundaries -- this is what lets convergence,
    oscillation, drift and abrupt change be told apart between snapshots.

    Two properties this must have, and neither is automatic:

    1. The measurement is COUNTERFACTUAL. It goes through
       `agent.evaluate()`, which has no side effects, so the probe set
       never becomes part of a recurrent agent's experience.
    2. Both terms are evaluated under the SAME conditioning state, so
       D_t reflects a change of parameters and not a change of internal
       state. That requires re-evaluating the PREVIOUS parameters under
       the CURRENT state, which is why the caller supplies a restorable
       snapshot of the previous parameters rather than cached
       probabilities from last trial.
    """

    def __init__(self, reference_set: list[np.ndarray]):
        self.reference_set = reference_set
        self._previous_params: dict | None = None

    def step(self, agent) -> float | None:
        """D_t comparing `agent` now against its parameters one step ago,
        both evaluated from the agent's current conditioning state."""
        conditioning = agent.recurrent_state()
        current = agent.evaluate(self.reference_set, conditioning_state=conditioning)

        previous_params = self._previous_params
        live_params = agent.parameters()
        self._previous_params = live_params

        if previous_params is None:
            return None

        # Swap parameters only -- recurrent and learner state are untouched,
        # so this temporary evaluation cannot perturb the trajectory.
        agent.set_parameters(previous_params)
        try:
            previous = agent.evaluate(self.reference_set, conditioning_state=conditioning)
        finally:
            agent.set_parameters(live_params)

        return float(np.mean([js_distance(p, q) for p, q in zip(previous, current)]))


def policy_distance(
    snapshot_i: PolicySnapshot,
    snapshot_j: PolicySnapshot,
    reference_set: list[np.ndarray],
) -> dict[str, float]:
    """Primary metric: mean Jensen-Shannon distance over S_ref. Secondary
    (also reported for transparency): mean total-variation distance."""
    weights_i, bias_i = np.array(snapshot_i.weights), np.array(snapshot_i.bias)
    weights_j, bias_j = np.array(snapshot_j.weights), np.array(snapshot_j.bias)

    js_values = []
    tv_values = []
    for features in reference_set:
        probs_i = _softmax_probs(weights_i, bias_i, features)
        probs_j = _softmax_probs(weights_j, bias_j, features)
        js_values.append(js_distance(probs_i, probs_j))
        tv_values.append(tv_distance_arrays(probs_i, probs_j))

    return {"js": float(np.mean(js_values)), "tv": float(np.mean(tv_values))}


def policy_distance_from_snapshots(
    snapshot_i: PolicySnapshot, snapshot_j: PolicySnapshot
) -> dict[str, float]:
    """Same primary/secondary metrics as policy_distance(), but computed
    from each snapshot's already-persisted reference_action_probs instead
    of reconstructing probabilities from weights/bias via a linear
    softmax. Works regardless of which agent architecture produced the
    snapshot -- unlike policy_distance(), which assumes
    logits = weights @ features + bias and must only ever be used with
    BaselineAgent snapshots."""
    js_values = []
    tv_values = []
    for key in snapshot_i.reference_action_probs:
        probs_i = np.array(snapshot_i.reference_action_probs[key])
        probs_j = np.array(snapshot_j.reference_action_probs[key])
        js_values.append(js_distance(probs_i, probs_j))
        tv_values.append(tv_distance_arrays(probs_i, probs_j))

    return {"js": float(np.mean(js_values)), "tv": float(np.mean(tv_values))}
