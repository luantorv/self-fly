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
