from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trial:
    trial_index: int
    stage: int
    stimulus_id: str
    category_id: str
    features: tuple[float, ...]
    action: str
    action_probs: tuple[float, float, float]
    reward: float
    ground_truth_label: str
    policy_weight_norm: float
    timestamp: str
    # JS(pi_t, pi_{t-1}) over the reference set, evaluated every trial.
    # None on the very first trial (no previous policy to compare against).
    # Added after every other field, with a default, so kwargs-based Trial
    # construction predating this field keeps working unchanged.
    policy_js_delta: float | None = None
    # Whether this trial fell on an experimental slot. Distinct from "the
    # stimulus was special": in BASELINE the slots are filled with ordinary
    # REAL/FALSA, and they still have to mark the same moments in the
    # timeline, otherwise BASELINE could never capture pi_1_pre/pi_1_post
    # and could never conclude stage 1 by stability at all.
    is_slot: bool = False


@dataclass
class PolicySnapshot:
    name: str
    stage: int
    trial_index: int
    weights: list
    bias: list
    reference_action_probs: dict
    timestamp: str
    # Complete agent state (every trainable layer, recurrent state, learner
    # state). `weights`/`bias` above are an inspection-oriented summary and
    # are NOT sufficient to reconstruct a recurrent agent; this field is.
    # Added last, with a default, so existing kwargs construction is unaffected.
    agent_state: dict = field(default_factory=dict)
