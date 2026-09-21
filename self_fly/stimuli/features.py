from __future__ import annotations

from dataclasses import dataclass

FEATURE_ORDER = (
    "shape_roundness",
    "color_hue",
    "color_saturation",
    "orientation_sin",
    "orientation_cos",
    "texture_density",
    "movement_speed",
    "movement_direction",
    "aspect_ratio",
    "context_level",
)


@dataclass(frozen=True)
class FeatureVector:
    shape_roundness: float
    color_hue: float
    color_saturation: float
    orientation_sin: float
    orientation_cos: float
    texture_density: float
    movement_speed: float
    movement_direction: float
    aspect_ratio: float
    context_level: float

    def to_tuple(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in FEATURE_ORDER)

    @classmethod
    def from_tuple(cls, values: tuple[float, ...]) -> "FeatureVector":
        return cls(**dict(zip(FEATURE_ORDER, values)))
