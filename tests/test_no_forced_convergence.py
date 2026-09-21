from dataclasses import replace

from self_fly.config.defaults import default_config
from self_fly.config.schema import StabilityConfig
from self_fly.experiment.engine import ExperimentEngine


def test_bare_pi_2_never_appears_when_stage_1_times_out():
    """Mechanical guardrail for the user's explicit requirement: a stage
    that never actually stabilizes must never be represented as if it had
    -- 'pi_2' bare is reserved for a real StageOutcome.STABLE."""
    config = default_config(seed=0)
    fast_stability = StabilityConfig(
        window_size=15,
        tv_threshold=0.35,
        reward_delta_threshold=0.6,
        cv_threshold=1.5,
        consistency_threshold=0.1,
        consecutive_windows_required=2,
    )
    stage0, stage1 = config.stages
    stage1 = replace(stage1, max_trials=5)  # far too small to ever stabilize
    config = replace(config, stability=fast_stability, stages=[stage0, stage1])

    engine = ExperimentEngine(config)
    max_steps = 20000
    steps_taken = 0
    while not engine.stage_machine.finished and steps_taken < max_steps:
        engine.step()
        steps_taken += 1

    assert engine.stage_machine.finished
    assert "pi_2" not in engine.policy_snapshots
    final_name = next(name for name in engine.policy_snapshots if name.startswith("pi_2"))
    assert final_name in ("pi_2_unstable", "pi_2_timeout")

    outcome_events = [e for e in engine.stage_events if e.kind == "stage_outcome"]
    assert len(outcome_events) == 1
    assert outcome_events[0].payload["outcome"] in ("unstable", "timeout")
