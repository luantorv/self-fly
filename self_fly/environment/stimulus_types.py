from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from self_fly.stimuli.features import FeatureVector


class StimulusLabel(Enum):
    REAL = "real"
    FALSA = "falsa"
    SELF_A = "self_a"
    SELF_B = "self_b"
    OTHER = "other"
    CONTROL = "control"
    # In-distribution like CONTROL and zero-reward like every special
    # category, but re-drawn on every appearance instead of frozen. That
    # single difference is what separates "the slot carries no reward"
    # from "the slot carries the same stimulus over and over".
    FRESH = "fresh"


@dataclass(frozen=True)
class GroundTruthStimulus:
    """Full record of a stimulus, including its privileged label.

    Lives only in experiment/ and environment/. Never crosses into agent/.
    """

    stimulus_id: str
    label: StimulusLabel
    features: FeatureVector
    category_id: str
    stage: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ObservableStimulus:
    """The only stimulus representation the agent ever receives.

    Deliberately has no label/category/stage field: there is nothing to
    strip at the call site because the type itself cannot carry it.
    """

    features: FeatureVector
