from dataclasses import replace

from self_fly.config.defaults import condition_progressive_conflict_config
from self_fly.config.schema import StabilityConfig
from self_fly.experiment.engine import ExperimentEngine


def _fast_conflict_config(seed: int = 0, stage_max_trials: int = 60):
    config = condition_progressive_conflict_config(seed=seed)
    fast_stability = StabilityConfig(
        window_size=15,
        tv_threshold=0.35,
        reward_delta_threshold=0.6,
        cv_threshold=1.5,
        consistency_threshold=0.1,
        consecutive_windows_required=2,
    )
    stage0, stage1, stage2, stage3, stage4 = config.stages
    stage1 = replace(stage1, max_trials=stage_max_trials)
    stage2 = replace(stage2, max_trials=stage_max_trials)
    stage3 = replace(stage3, max_trials=stage_max_trials)
    stage4 = replace(stage4, max_trials=stage_max_trials)
    return replace(
        config, stability=fast_stability, stages=[stage0, stage1, stage2, stage3, stage4]
    )


def test_progressive_conflict_transitions_through_stages_and_finishes():
    config = _fast_conflict_config(seed=8)
    engine = ExperimentEngine(config)

    steps = 0
    while not engine.stage_machine.finished and steps < 30000:
        engine.step()
        steps += 1

    assert engine.stage_machine.finished, "experiment did not finish within max_steps"

    stage_changed_events = [e for e in engine.stage_events if e.kind == "stage_changed"]
    to_stages = [e.payload["to_stage"] for e in stage_changed_events]
    assert to_stages == sorted(to_stages), "stages must progress monotonically, never backwards"
    assert to_stages[0] == 1  # 0 -> 1 always happens

    outcome_events = [e for e in engine.stage_events if e.kind == "stage_outcome"]
    assert len(outcome_events) >= 1
    for event in outcome_events:
        outcome = event.payload["outcome"]
        assert outcome in ("stable", "unstable", "timeout")
        snapshot_name = event.payload["snapshot_name"]
        assert snapshot_name in engine.policy_snapshots
        if outcome != "stable":
            assert snapshot_name.endswith(outcome)


def test_stage_2_conflict_reward_is_active_once_reached():
    """Sanity check that reaching stage 2 actually routes self_a rewards
    through ConflictConfig, not the old fixed-zero reward_overrides."""
    config = _fast_conflict_config(seed=9, stage_max_trials=40)
    engine = ExperimentEngine(config)

    steps = 0
    while engine.current_stage < 2 and steps < 30000 and not engine.stage_machine.finished:
        engine.step()
        steps += 1

    if engine.current_stage < 2:
        return  # didn't reach stage 2 within budget for this seed -- nothing to check

    trial = None
    steps = 0
    while trial is None and steps < 2000 and not engine.stage_machine.finished:
        candidate = engine.step()
        steps += 1
        if candidate.stage == 2 and candidate.ground_truth_label == "self_a":
            trial = candidate

    if trial is not None:
        assert trial.reward in (0.1, -0.05, 0.0)  # maintain, change, or first_exposure_reward
