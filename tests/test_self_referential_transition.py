from dataclasses import replace

from self_fly.config.defaults import default_config
from self_fly.config.schema import StabilityConfig
from self_fly.experiment.engine import ExperimentEngine


def _fast_stability_config(seed: int = 0):
    config = default_config(seed=seed)
    fast_stability = StabilityConfig(
        window_size=15,
        tv_threshold=0.35,
        reward_delta_threshold=0.6,
        cv_threshold=1.5,
        consistency_threshold=0.1,
        consecutive_windows_required=2,
    )
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
    assert set(engine.policy_snapshots.keys()) == {"pi_0", "pi_1", "pi_2"}

    pi0 = engine.policy_snapshots["pi_0"]
    pi1 = engine.policy_snapshots["pi_1"]
    pi2 = engine.policy_snapshots["pi_2"]

    assert pi0.stage == 0
    assert pi1.stage == 1
    assert pi2.stage == 1
    assert pi0.trial_index < pi1.trial_index < pi2.trial_index

    event_kinds = [e.kind for e in engine.stage_events]
    assert event_kinds.count("stage_changed") == 1
    assert event_kinds.index("capture_snapshot") < event_kinds.index("stage_changed")
    assert "capture_snapshot" in event_kinds
    assert event_kinds[-1] == "experiment_finished"

    stage_changed_events = [e for e in engine.stage_events if e.kind == "stage_changed"]
    assert stage_changed_events[0].payload == {"from_stage": 0, "to_stage": 1}


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
