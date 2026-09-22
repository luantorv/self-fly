from __future__ import annotations

import numpy as np

from self_fly.config.schema import ProtocolConfig, SelfVariantConfig
from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel

from .self_stimulus import SelfStimulusGenerator


def neutral_signature(
    n_features: int, variant: SelfVariantConfig, protocol: ProtocolConfig
) -> np.ndarray:
    """The interpolation target for SELF(eps): a signature belonging to no
    trained agent at all.

    Drawn from the protocol seed, so it is the same vector for every
    experimental seed, condition and architecture -- SELF(eps) then differs
    across runs only because the agent's own signature differs, never
    because the target moved. Magnitudes are positive because a signature
    is a vector of per-feature norms.
    """
    rng = np.random.default_rng(protocol.protocol_seed + variant.neutral_signature_seed_offset)
    return np.abs(rng.normal(0.0, 1.0, size=n_features))


def interpolate_signature(
    own: np.ndarray, target: np.ndarray, epsilon: float
) -> np.ndarray:
    """s_eps = (1-eps) * s_own + eps * s_target.

    In SIGNATURE space, before the feature transformation -- never between
    already-transformed stimuli. Interpolating transformed feature vectors
    and re-applying the transformation would apply the signed-feature
    remapping twice and push those features out of range.
    """
    return (1.0 - epsilon) * own + epsilon * target


def build_self_epsilon(
    generator: SelfStimulusGenerator,
    own_signature: np.ndarray,
    target_signature: np.ndarray,
    target_distance: float,
    variant: SelfVariantConfig,
    stage: int,
    n_grid: int = 2001,
) -> tuple[GroundTruthStimulus, dict]:
    """SELF(eps*): the variant of the agent's own signature that sits as
    far from SELF_A, in stimulus space, as OTHER does.

    eps* is found on a grid using a purely GEOMETRIC distance over the
    feature vector. Matching on a response-space quantity (JS, TV,
    entropy) would be circular -- it would tune the control using the very
    dependent variable the control exists to interpret.
    """
    to_features = lambda sig: np.array(
        generator.generate(sig, stage=stage, label=StimulusLabel.SELF_B).features.to_tuple()
    )
    own_features = to_features(own_signature)

    best_epsilon, best_error, best_distance = 0.0, float("inf"), 0.0
    for epsilon in np.linspace(0.0, 1.0, n_grid):
        candidate = to_features(interpolate_signature(own_signature, target_signature, epsilon))
        distance = float(np.linalg.norm(own_features - candidate))
        error = abs(distance - target_distance)
        if error < best_error:
            best_epsilon, best_error, best_distance = float(epsilon), error, distance

    relative_error = best_error / target_distance if target_distance > 0 else float("inf")
    stimulus = generator.generate(
        interpolate_signature(own_signature, target_signature, best_epsilon),
        stage=stage,
        label=StimulusLabel.SELF_B,
    )
    diagnostics = {
        "epsilon": best_epsilon,
        "achieved_distance": best_distance,
        "target_distance": target_distance,
        "relative_error": relative_error,
        "matched": relative_error <= variant.epsilon_match_tolerance,
        "distance_metric": "euclidean_feature_space",
    }
    return stimulus, diagnostics
