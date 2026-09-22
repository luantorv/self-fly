from __future__ import annotations

import itertools

import numpy as np

from self_fly.agent.factory import make_agent
from self_fly.config.schema import ExperimentConfig

from .entropy import action_entropy
from .policy_distance import js_distance, tv_distance_arrays
from .types import PolicySnapshot


def evaluate_snapshots_on_special_stimuli(
    config: ExperimentConfig,
    snapshots: dict[str, PolicySnapshot],
    special_stimuli: dict[str, np.ndarray],
) -> dict:
    """How each captured policy responds to each frozen special stimulus,
    plus the pairwise distances between those responses.

    This answers the question that needs no exposure at all: does the
    policy already treat a stimulus built from its own weights differently
    from a structurally equivalent one built from another agent's? Because
    it needs no exposure, it can be measured in EVERY condition -- and at
    pi_0, which is captured before SELF_A even exists. That is why this
    runs post-hoc over the stored snapshots (each carries the full agent
    state) rather than inline during the run.

    The pairwise distances are the primary measurement of the whole
    experiment; before this existed they had to be recomputed by hand from
    the logs, which is exactly what the protocol forbids.
    """
    if not special_stimuli:
        return {}

    names = sorted(special_stimuli)
    inputs = np.array([special_stimuli[n] for n in names])

    out: dict = {}
    for snapshot_name, snapshot in snapshots.items():
        if not snapshot.agent_state:
            continue

        agent = make_agent(
            config.agent_type, config.learner, config.connectome, seed=config.seed + 3
        )
        agent.load_state(snapshot.agent_state)
        probs = agent.evaluate(inputs, conditioning_state=agent.recurrent_state())

        by_name = {name: probs[i] for i, name in enumerate(names)}
        entry = {
            "action_probs": {n: p.tolist() for n, p in by_name.items()},
            "entropy": {n: action_entropy(p) for n, p in by_name.items()},
            "argmax_action": {n: int(p.argmax()) for n, p in by_name.items()},
            "pairwise": {},
        }
        for a, b in itertools.combinations(names, 2):
            entry["pairwise"][f"{a}|{b}"] = {
                "js": js_distance(by_name[a], by_name[b]),
                "tv": tv_distance_arrays(by_name[a], by_name[b]),
            }
        out[snapshot_name] = entry

    return out


def stimulus_space_distances(special_stimuli: dict[str, np.ndarray]) -> dict:
    """Pairwise Euclidean distances between the frozen stimuli themselves.

    Reported alongside the response distances because they are the null
    any response difference has to be read against: if SELF_A and a
    variant sit far closer together in stimulus space than either does to
    OTHER, then their responses being closer is arithmetic, not a finding.
    """
    names = sorted(special_stimuli)
    return {
        f"{a}|{b}": float(np.linalg.norm(special_stimuli[a] - special_stimuli[b]))
        for a, b in itertools.combinations(names, 2)
    }
