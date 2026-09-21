from self_fly.config.schema import StabilityConfig, StageConfig
from self_fly.experiment.stability import StabilityDetector
from self_fly.experiment.stage_outcome import StageOutcome
from self_fly.experiment.stages import Stage, StageMachine
from self_fly.experiment.types import Trial


def _trial(idx: int, action: str, reward: float, label: str = "real") -> Trial:
    return Trial(
        trial_index=idx,
        stage=1,
        stimulus_id=f"s{idx}",
        category_id="c0",
        features=tuple([0.0] * 10),
        action=action,
        action_probs=(0.33, 0.33, 0.34),
        reward=reward,
        ground_truth_label=label,
        policy_weight_norm=0.0,
        timestamp="t",
    )


def _stage1_config(max_trials: int) -> StageConfig:
    return StageConfig(
        stage=1,
        stimulus_mix={"real": 1.0},
        reward_overrides={},
        entry_condition="stability_detected",
        exit_condition="stability_detected_or_max_trials",
        max_trials=max_trials,
    )


def _machine_in_stage1(stability_config: StabilityConfig, max_trials: int):
    detector = StabilityDetector(stability_config)
    machine = StageMachine([_stage1_config(max_trials)], stability_detector=detector)
    machine.current_stage = Stage.STAGE_1_SELF_INTRODUCED
    return detector, machine


def test_timeout_when_no_partial_stability_was_ever_observed():
    stability_config = StabilityConfig(
        window_size=10,
        tv_threshold=0.15,
        reward_delta_threshold=0.2,
        cv_threshold=0.35,
        consistency_threshold=0.6,
        consecutive_windows_required=3,
    )
    detector, machine = _machine_in_stage1(stability_config, max_trials=1)

    trial = _trial(0, "REAL", 1.0, label="self_a")
    event = detector.update(trial)
    events = machine.process(trial, event)

    assert detector.max_consecutive_stable_observed == 0
    outcomes = [e for e in events if e.kind == "stage_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0].payload["outcome"] == StageOutcome.TIMEOUT.value

    snapshots = [e for e in events if e.kind == "capture_snapshot"]
    # pi_1 (first self exposure) and the stage's own conclusion snapshot.
    final_snapshot = [s for s in snapshots if s.payload["name"] != "pi_1"][0]
    assert final_snapshot.payload["name"] == "pi_2_timeout"
    assert "pi_2" not in [s.payload["name"] for s in snapshots]


def test_unstable_when_partial_stability_breaks_before_budget_runs_out():
    stability_config = StabilityConfig(
        window_size=10,
        tv_threshold=0.15,
        reward_delta_threshold=0.2,
        cv_threshold=0.35,
        consistency_threshold=0.6,
        consecutive_windows_required=3,
    )
    detector, machine = _machine_in_stage1(stability_config, max_trials=45)

    finished_events: list = []
    idx = 0

    # First trial exposes self_a so hit_stability can ever count from here on.
    trial = _trial(idx, "REAL", 1.0, label="self_a")
    event = detector.update(trial)
    machine.process(trial, event)
    idx += 1

    # Two more fully-consistent windows (29 more trials, deterministic: same
    # action/reward every time) -> two consecutive stable comparisons, one
    # short of the three required to actually fire stability.
    while idx < 30:
        trial = _trial(idx, "REAL", 1.0, label="real")
        event = detector.update(trial)
        machine.process(trial, event)
        idx += 1
    assert detector.max_consecutive_stable_observed == 2

    # Abrupt, sustained change: breaks the streak and never lets it rebuild
    # before the trial budget (max_trials=45) runs out.
    while idx < 45 and not machine.finished:
        trial = _trial(idx, "FALSA", -1.0, label="real")
        event = detector.update(trial)
        finished_events = machine.process(trial, event)
        idx += 1

    assert machine.finished
    outcomes = [e for e in finished_events if e.kind == "stage_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0].payload["outcome"] == StageOutcome.UNSTABLE.value
    snapshots = [e for e in finished_events if e.kind == "capture_snapshot"]
    assert snapshots[0].payload["name"] == "pi_2_unstable"


def test_stable_outcome_uses_bare_pi_2_name():
    stability_config = StabilityConfig(
        window_size=10,
        tv_threshold=0.15,
        reward_delta_threshold=0.2,
        cv_threshold=0.35,
        consistency_threshold=0.6,
        consecutive_windows_required=2,
    )
    detector, machine = _machine_in_stage1(stability_config, max_trials=1000)

    idx = 0
    trial = _trial(idx, "REAL", 1.0, label="self_a")
    event = detector.update(trial)
    events = machine.process(trial, event)
    idx += 1

    while not machine.finished and idx < 100:
        trial = _trial(idx, "REAL", 1.0, label="real")
        event = detector.update(trial)
        events = machine.process(trial, event)
        idx += 1

    assert machine.finished
    outcomes = [e for e in events if e.kind == "stage_outcome"]
    assert outcomes[0].payload["outcome"] == StageOutcome.STABLE.value
    snapshots = [e for e in events if e.kind == "capture_snapshot"]
    assert snapshots[0].payload["name"] == "pi_2"
