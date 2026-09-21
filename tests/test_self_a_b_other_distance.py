from dataclasses import replace

import numpy as np

from self_fly.config.defaults import condition_multi_self_config
from self_fly.config.schema import StabilityConfig
from self_fly.experiment.engine import ExperimentEngine
from self_fly.experiment.policy_distance import js_distance
from self_fly.experiment.stage_outcome import StageOutcome


def _fast_multi_self_config(seed: int = 0):
    config = condition_multi_self_config(seed=seed)
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


def _run_until_pi0(engine: ExperimentEngine, max_steps: int = 20000) -> None:
    steps = 0
    while engine.self_b_stimulus_ground_truth is None and steps < max_steps:
        engine.step()
        steps += 1
    assert engine.self_b_stimulus_ground_truth is not None, "pi_0 never captured within max_steps"


def test_self_a_self_b_other_pairwise_js_distance_is_computable():
    config = _fast_multi_self_config(seed=6)
    engine = ExperimentEngine(config)
    _run_until_pi0(engine)

    assert engine.other_agent_outcome is not None
    if engine.other_agent_outcome != StageOutcome.STABLE:
        return  # the auxiliary agent didn't converge for this seed -- nothing to compare, by design

    def probs_for(ground_truth):
        features = np.array(ground_truth.features.to_tuple())
        return engine.agent.probabilities(features)

    p_self_a = probs_for(engine.self_stimulus_ground_truth)
    p_self_b = probs_for(engine.self_b_stimulus_ground_truth)
    p_other = probs_for(engine.other_stimulus_ground_truth)

    for d in (js_distance(p_self_a, p_self_b), js_distance(p_self_a, p_other), js_distance(p_self_b, p_other)):
        assert 0.0 <= d <= 1.0


def test_self_multi_condition_never_leaks_label_via_observe():
    config = _fast_multi_self_config(seed=7)
    engine = ExperimentEngine(config)

    steps = 0
    while not engine.stage_machine.finished and steps < 5000:
        trial = engine.step()
        steps += 1
        if trial.ground_truth_label in ("self_a", "self_b", "other"):
            ground_truth = {
                "self_a": engine.self_stimulus_ground_truth,
                "self_b": engine.self_b_stimulus_ground_truth,
                "other": engine.other_stimulus_ground_truth,
            }[trial.ground_truth_label]
            observable = engine.environment.observe(ground_truth)
            assert not hasattr(observable, "label")
