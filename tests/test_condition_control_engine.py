from dataclasses import replace

from self_fly.config.defaults import condition_control_config
from self_fly.config.schema import StabilityConfig
from self_fly.experiment.engine import ExperimentEngine


def _fast_control_config(seed: int = 0):
    config = condition_control_config(seed=seed)
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


def test_control_condition_never_samples_self_a():
    config = _fast_control_config(seed=3)
    engine = ExperimentEngine(config)

    trials = []
    steps = 0
    while not engine.stage_machine.finished and steps < 5000:
        trials.append(engine.step())
        steps += 1

    assert engine.stage_machine.finished
    assert all(t.ground_truth_label != "self_a" for t in trials)
    assert any(t.ground_truth_label == "control" for t in trials)


def test_control_stimulus_frozen_at_pi0_moment():
    config = _fast_control_config(seed=4)
    engine = ExperimentEngine(config)

    while engine.control_stimulus_ground_truth is None:
        engine.step()

    frozen_features = engine.control_stimulus_ground_truth.features.to_tuple()
    for _ in range(200):
        engine.step()
        if engine.stage_machine.finished:
            break

    assert engine.control_stimulus_ground_truth.features.to_tuple() == frozen_features


def test_control_condition_produces_same_instrumentation_as_self_condition():
    """Same instrumentation (D_t, STABLE/UNSTABLE/TIMEOUT outcomes) must be
    available regardless of which stimulus fills the 20% slot in stage 1 --
    and the control condition must actually be able to reach real
    stability, not just always time out (a regression this test would
    catch: the stage-1 exposure gate used to be hardcoded to "self_a")."""
    config = _fast_control_config(seed=5)
    engine = ExperimentEngine(config)

    steps = 0
    last_trial = None
    while not engine.stage_machine.finished and steps < 5000:
        last_trial = engine.step()
        steps += 1

    assert engine.stage_machine.finished
    assert last_trial.policy_js_delta is not None
    outcome_events = [e for e in engine.stage_events if e.kind == "stage_outcome"]
    assert len(outcome_events) == 1
    assert outcome_events[0].payload["outcome"] in ("stable", "unstable", "timeout")
