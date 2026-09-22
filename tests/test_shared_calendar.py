from dataclasses import replace

import numpy as np

from self_fly.config.conditions import CONDITION_BUILDERS, SLOT_FRACTION
from self_fly.experiment.engine import ExperimentEngine
from stability_helpers import fast_stability_config

LADDER = ["baseline", "fresh", "control", "self"]


def _run(name, seed=0, n=900):
    config = CONDITION_BUILDERS[name](seed=seed)
    stage0, stage1 = config.stages
    config = replace(
        config,
        stability=fast_stability_config(),
        stages=[stage0, replace(stage1, max_trials=4000)],
    )
    return ExperimentEngine(config).run(n)


def test_slot_indices_are_identical_across_conditions():
    """Every condition shares one experimental calendar: slot k falls on
    the same trial index everywhere, so the conditions differ only in what
    fills a slot -- not in when slots happen."""
    slots = {name: [t.trial_index for t in _run(name) if t.is_slot] for name in LADDER}
    reference = slots["baseline"]

    assert len(reference) > 50, "too few slots to be a meaningful check"
    for name in LADDER:
        assert slots[name] == reference, name


def test_non_slot_stimuli_are_identical_across_conditions():
    """Ordinary REAL/FALSA trials come from their own stream, so enabling
    a special stimulus cannot shift the ordinary content. Previously a
    single RNG served everything and the conditions diverged from the very
    first draw."""
    runs = {name: _run(name) for name in LADDER}
    reference = runs["baseline"]

    for name in LADDER[1:]:
        compared = 0
        for a, b in zip(reference, runs[name]):
            if a.is_slot:
                continue
            assert np.allclose(a.features, b.features), (name, a.trial_index)
            assert a.ground_truth_label == b.ground_truth_label
            compared += 1
        assert compared > 100, (name, compared)


def test_slot_fraction_is_respected_and_equal_everywhere():
    for name in LADDER:
        trials = [t for t in _run(name) if t.stage == 1]
        observed = sum(t.is_slot for t in trials) / len(trials)
        assert abs(observed - SLOT_FRACTION) < 0.06, (name, observed)


def test_baseline_fills_slots_with_ordinary_stimuli():
    """BASELINE keeps the slots but puts ordinary rewarded REAL/FALSA in
    them, which is what keeps its amount of ordinary training comparable
    while still giving it a matched moment for pi_1_pre/pi_1_post."""
    trials = _run("baseline")
    slot_trials = [t for t in trials if t.is_slot]

    assert slot_trials
    assert all(t.ground_truth_label in ("real", "falsa") for t in slot_trials)


def test_fresh_slots_are_never_the_same_stimulus_twice():
    """FRESH is the one special category that is not frozen -- that single
    difference is what separates 'no reward' from 'the same stimulus over
    and over'."""
    trials = [t for t in _run("fresh") if t.is_slot and t.ground_truth_label == "fresh"]
    assert len(trials) > 20

    features = {tuple(t.features) for t in trials}
    assert len(features) == len(trials)
    assert all(t.reward == 0.0 for t in trials)


def test_control_slots_are_always_the_same_frozen_stimulus():
    trials = [t for t in _run("control") if t.ground_truth_label == "control"]
    assert len(trials) > 20

    features = {tuple(t.features) for t in trials}
    assert len(features) == 1
    assert all(t.reward == 0.0 for t in trials)


def test_baseline_can_reach_stability_and_capture_both_pi1_snapshots():
    """Regression guard: while the gating keyed on 'the stimulus is
    special', BASELINE could never set that flag, so stage 1 was
    structurally incapable of concluding STABLE and never captured pi_1 --
    the one condition meant to serve as the null was unusable."""
    config = CONDITION_BUILDERS["baseline"](seed=0)
    stage0, stage1 = config.stages
    config = replace(
        config,
        stability=fast_stability_config(),
        stages=[stage0, replace(stage1, max_trials=2000)],
    )
    engine = ExperimentEngine(config)

    steps = 0
    while not engine.stage_machine.finished and steps < 20000:
        engine.step()
        steps += 1

    assert engine.stage_machine.finished
    assert {"pi_0", "pi_1_pre", "pi_1_post"} <= set(engine.policy_snapshots)
    outcomes = [e.payload["outcome"] for e in engine.stage_events if e.kind == "stage_outcome"]
    assert outcomes == ["stable"]
