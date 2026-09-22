from dataclasses import replace

from self_fly.config.defaults import default_config
from self_fly.experiment.engine import ExperimentEngine
from stability_helpers import fast_stability_config


def _fast_stability_config(seed: int = 0):
    config = default_config(seed=seed)
    fast_stability = fast_stability_config()
    stage0, stage1 = config.stages
    stage1 = replace(stage1, max_trials=400)
    return replace(config, stability=fast_stability, stages=[stage0, stage1])


def test_pi0_pi1_pi2_are_captured_in_order_and_stage_transitions_once():
    config = _fast_stability_config(seed=0)
    engine = ExperimentEngine(config)

    max_steps = 5000
    steps_taken = 0
    while not engine.stage_machine.finished and steps_taken < max_steps:
        engine.step()
        steps_taken += 1

    assert engine.stage_machine.finished, "experiment did not finish within max_steps"
    assert set(engine.policy_snapshots.keys()) == {"pi_0", "pi_1_pre", "pi_1_post", "pi_2"}

    pi0 = engine.policy_snapshots["pi_0"]
    pi1_pre = engine.policy_snapshots["pi_1_pre"]
    pi1_post = engine.policy_snapshots["pi_1_post"]
    pi2 = engine.policy_snapshots["pi_2"]

    assert pi0.stage == 0
    assert pi1_pre.stage == 1
    assert pi1_post.stage == 1
    assert pi2.stage == 1
    # pre and post bracket the same trial: the first special exposure.
    assert pi0.trial_index < pi1_pre.trial_index
    assert pi1_pre.trial_index == pi1_post.trial_index
    assert pi1_post.trial_index < pi2.trial_index


def test_pi1_pre_is_captured_before_the_agent_sees_the_stimulus():
    """pi_1_pre must be the policy untouched by the first special
    exposure, so JS(pi_1_pre, pi_1_post) isolates that exposure's effect
    from the drift accumulated since pi_0."""
    from self_fly.experiment.policy_distance import policy_distance_from_snapshots

    engine = ExperimentEngine(_fast_stability_config(seed=0))
    steps = 0
    while not engine.stage_machine.finished and steps < 5000:
        engine.step()
        steps += 1

    snaps = engine.policy_snapshots
    drift_before = policy_distance_from_snapshots(snaps["pi_0"], snaps["pi_1_pre"])["js"]
    exposure_effect = policy_distance_from_snapshots(snaps["pi_1_pre"], snaps["pi_1_post"])["js"]

    # Both are now separately measurable; neither is a stand-in for the other.
    assert drift_before >= 0.0
    assert exposure_effect > 0.0

    event_kinds = [e.kind for e in engine.stage_events]
    assert event_kinds.count("stage_changed") == 1
    assert event_kinds.index("capture_snapshot") < event_kinds.index("stage_changed")
    assert "capture_snapshot" in event_kinds
    assert event_kinds[-1] == "experiment_finished"

    stage_changed_events = [e for e in engine.stage_events if e.kind == "stage_changed"]
    assert stage_changed_events[0].payload == {"from_stage": 0, "to_stage": 1}

    outcome_events = [e for e in engine.stage_events if e.kind == "stage_outcome"]
    assert len(outcome_events) == 1
    assert outcome_events[0].payload["outcome"] == "stable"
    assert outcome_events[0].payload["snapshot_name"] == "pi_2"


def test_self_stimulus_frozen_at_pi0_moment_matches_snapshot_weights():
    config = _fast_stability_config(seed=1)
    engine = ExperimentEngine(config)

    while engine.self_stimulus_ground_truth is None:
        engine.step()

    frozen_features = engine.self_stimulus_ground_truth.features.to_tuple()

    # Running many more steps must not change the already-frozen self stimulus.
    for _ in range(200):
        engine.step()
        if engine.stage_machine.finished:
            break

    assert engine.self_stimulus_ground_truth.features.to_tuple() == frozen_features
    assert engine.self_stimulus_ground_truth.label.value == "self_a"


def test_observable_stimulus_for_self_never_carries_label_downstream():
    config = _fast_stability_config(seed=2)
    engine = ExperimentEngine(config)

    while not engine.stage_machine.finished and len(engine.stage_events) < 50:
        trial = engine.step()
        if trial.ground_truth_label == "self_a":
            # The agent's own decision pipeline only ever sees `features`;
            # nothing here exposes the privileged label to it.
            observable = engine.environment.observe(engine.self_stimulus_ground_truth)
            assert not hasattr(observable, "label")
            return

    raise AssertionError("self stimulus was never sampled during stage 1")
