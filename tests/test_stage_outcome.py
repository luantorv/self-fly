from self_fly.config.schema import StabilityConfig, StageConfig
from self_fly.experiment.stability import StabilityDetector
from self_fly.experiment.stage_outcome import StageOutcome
from self_fly.experiment.stages import Stage, StageMachine
from self_fly.experiment.types import Trial

# Stability now has two conjunctive components: task accuracy and median
# D_t. These tests drive the second directly and hold the first at a
# constant pass (accuracy_fn=None -> 1.0), so each test isolates exactly
# one mechanism: the outcome that a stage reports when it concludes.
SETTLED = 0.0001  # well under max_median_policy_delta
DRIFTING = 0.5  # well over it


def _trial(
    idx: int, action: str, reward: float, ground_truth: str = "real", is_slot: bool = False
) -> Trial:
    return Trial(
        trial_index=idx,
        stage=1,
        stimulus_id=f"s{idx}",
        category_id="c0",
        features=tuple([0.0] * 10),
        action=action,
        action_probs=(0.33, 0.33, 0.34),
        reward=reward,
        ground_truth_label=ground_truth,
        policy_weight_norm=0.0,
        timestamp="t",
        is_slot=is_slot,
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


def _machine_in_stage1(max_trials: int, windows_required: int = 3):
    config = StabilityConfig(
        window_size=10,
        min_accuracy=0.0,
        max_median_policy_delta=0.002,
        consecutive_windows_required=windows_required,
    )
    detector = StabilityDetector(config)
    machine = StageMachine([_stage1_config(max_trials)], stability_detector=detector)
    machine.current_stage = Stage.STAGE_1_SELF_INTRODUCED
    return detector, machine


def _feed(detector, machine, start, count, delta, ground_truth="real"):
    events = []
    idx = start
    for _ in range(count):
        trial = _trial(idx, "REAL", 1.0, ground_truth=ground_truth)
        event = detector.update(trial, delta)
        events = machine.process(trial, event)
        idx += 1
        if machine.finished:
            break
    return idx, events


def test_timeout_when_no_partial_stability_was_ever_observed():
    detector, machine = _machine_in_stage1(max_trials=1)

    trial = _trial(0, "REAL", 1.0, ground_truth="self_a", is_slot=True)
    event = detector.update(trial, DRIFTING)
    events = machine.process(trial, event)

    assert detector.max_consecutive_stable_observed == 0
    outcomes = [e for e in events if e.kind == "stage_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0].payload["outcome"] == StageOutcome.TIMEOUT.value

    snapshots = [e for e in events if e.kind == "capture_snapshot"]
    final_snapshot = [s for s in snapshots if s.payload["name"] != "pi_1_post"][0]
    assert final_snapshot.payload["name"] == "pi_2_timeout"
    assert "pi_2" not in [s.payload["name"] for s in snapshots]


def test_unstable_when_partial_stability_breaks_before_budget_runs_out():
    detector, machine = _machine_in_stage1(max_trials=60, windows_required=3)

    # First trial is a special exposure so hit_stability can count later.
    trial = _trial(0, "REAL", 1.0, ground_truth="self_a", is_slot=True)
    machine.process(trial, detector.update(trial, DRIFTING))

    # Two settled windows -> a streak of 2, one short of the 3 required.
    idx, _ = _feed(detector, machine, 1, 20, SETTLED)
    assert detector.max_consecutive_stable_observed == 2
    assert not machine.finished

    # Sustained drift: breaks the streak and never lets it rebuild before
    # the trial budget runs out.
    _, events = _feed(detector, machine, idx, 60, DRIFTING)

    assert machine.finished
    outcomes = [e for e in events if e.kind == "stage_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0].payload["outcome"] == StageOutcome.UNSTABLE.value
    assert [e for e in events if e.kind == "capture_snapshot"][0].payload["name"] == "pi_2_unstable"


def test_stable_outcome_uses_bare_pi_2_name():
    detector, machine = _machine_in_stage1(max_trials=1000, windows_required=2)

    trial = _trial(0, "REAL", 1.0, ground_truth="self_a", is_slot=True)
    events = machine.process(trial, detector.update(trial, SETTLED))
    _, events = _feed(detector, machine, 1, 100, SETTLED)

    assert machine.finished
    outcomes = [e for e in events if e.kind == "stage_outcome"]
    assert outcomes[0].payload["outcome"] == StageOutcome.STABLE.value
    assert [e for e in events if e.kind == "capture_snapshot"][0].payload["name"] == "pi_2"


def test_accuracy_component_can_block_stability_on_its_own():
    """A perfectly settled policy that cannot do the task is not stable:
    the two components are conjunctive."""
    config = StabilityConfig(
        window_size=10,
        min_accuracy=0.70,
        max_median_policy_delta=0.002,
        consecutive_windows_required=2,
    )
    detector = StabilityDetector(config, accuracy_fn=lambda: 0.55)
    for idx in range(100):
        assert detector.update(_trial(idx, "REAL", 1.0), SETTLED) is None
    assert detector.max_consecutive_stable_observed == 0
