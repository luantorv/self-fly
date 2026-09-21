from __future__ import annotations

import itertools

import numpy as np

from self_fly.config.schema import StimulusConfig
from self_fly.environment.stimulus_types import GroundTruthStimulus, StimulusLabel

from .features import FEATURE_ORDER, FeatureVector

_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_ORDER)}
_SIGNED_FEATURES = ("orientation_sin", "orientation_cos", "movement_direction")


class StimulusGenerator:
    """Generates REAL/FALSA stimuli whose label depends on a weighted, noisy
    combination of several features, so that no single feature is a trivial
    shortcut for classification."""

    def __init__(self, config: StimulusConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._counter = itertools.count()
        self._weights = np.array(
            [config.signal_feature_weights.get(name, 0.0) for name in FEATURE_ORDER]
        )

    def _sample_raw_features(self) -> np.ndarray:
        raw = self.rng.uniform(0.0, 1.0, size=len(FEATURE_ORDER))
        for name in _SIGNED_FEATURES:
            raw[_FEATURE_INDEX[name]] = self.rng.uniform(-1.0, 1.0)
        return raw

    def _label_for(self, raw: np.ndarray) -> StimulusLabel:
        centered = raw - 0.5
        signal = float(np.dot(self._weights, centered))
        noise = self.rng.normal(0.0, self.config.noise_std)
        return StimulusLabel.REAL if (signal + noise) > 0 else StimulusLabel.FALSA

    def _category_id(self, raw: np.ndarray, label: StimulusLabel) -> str:
        grid = self.config.category_grid_size
        bucket_a = min(int(raw[_FEATURE_INDEX["shape_roundness"]] * grid), grid - 1)
        bucket_b = min(int(raw[_FEATURE_INDEX["color_hue"]] * grid), grid - 1)
        return f"{label.value}_c{bucket_a}{bucket_b}"

    def sample(self, stage: int) -> GroundTruthStimulus:
        raw = self._sample_raw_features()
        label = self._label_for(raw)
        category_id = self._category_id(raw, label)
        features = FeatureVector.from_tuple(tuple(raw.tolist()))
        stimulus_id = f"s{next(self._counter)}"
        return GroundTruthStimulus(
            stimulus_id=stimulus_id,
            label=label,
            features=features,
            category_id=category_id,
            stage=stage,
        )

    def sample_labeled(
        self, stage: int, label: StimulusLabel, max_attempts: int = 10_000
    ) -> GroundTruthStimulus:
        """Rejection-sample a REAL/FALSA stimulus of a specific label. Label
        frequency is close to 50/50 by construction, so this converges in a
        handful of attempts."""
        for _ in range(max_attempts):
            candidate = self.sample(stage)
            if candidate.label == label:
                return candidate
        raise RuntimeError(
            f"Could not sample a stimulus with label={label} after {max_attempts} attempts"
        )
