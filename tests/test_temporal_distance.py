import numpy as np

from self_fly.analysis.temporal_dynamics import classify_dt_series
from self_fly.config.defaults import default_config, condition_self_config
from self_fly.config.schema import ConnectomeConfig
from self_fly.agent.connectome.agent import ConnectomeInspiredAgent
from self_fly.experiment.engine import ExperimentEngine
from self_fly.experiment.policy_distance import TemporalDistanceTracker
from dataclasses import replace


class _FixedAgent:
    """Minimal Agent-shaped stub whose policy only changes when the test
    explicitly changes `probs`."""

    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float)

    def evaluate(self, inputs_batch, conditioning_state=None):
        return np.array([self.probs for _ in inputs_batch])

    def recurrent_state(self):
        return None

    def parameters(self):
        return {"probs": self.probs.copy()}

    def set_parameters(self, params):
        self.probs = params["probs"].copy()


def test_first_step_returns_none():
    tracker = TemporalDistanceTracker(reference_set=[np.zeros(10)])
    assert tracker.step(_FixedAgent([1 / 3, 1 / 3, 1 / 3])) is None


def test_step_returns_zero_when_policy_unchanged():
    tracker = TemporalDistanceTracker(reference_set=[np.zeros(10), np.ones(10)])
    agent = _FixedAgent([0.5, 0.3, 0.2])

    tracker.step(agent)
    assert tracker.step(agent) == 0.0


def test_step_returns_positive_after_a_real_change():
    tracker = TemporalDistanceTracker(reference_set=[np.zeros(10)])
    agent = _FixedAgent([1.0, 0.0, 0.0])
    tracker.step(agent)

    agent.probs = np.array([0.0, 1.0, 0.0])
    assert tracker.step(agent) > 0.0


def test_measuring_dt_never_mutates_a_recurrent_agent():
    """The central fix: for a recurrent agent a forward pass IS
    experience, so the probe set must never go through the live path."""
    agent = ConnectomeInspiredAgent(ConnectomeConfig(), seed=0)
    agent.probabilities(np.linspace(0, 1, 10))  # advance the live state once
    tracker = TemporalDistanceTracker([np.linspace(0, 1, 10) for _ in range(20)])

    before = agent.recurrent_state()
    tracker.step(agent)
    tracker.step(agent)
    after = agent.recurrent_state()

    assert np.array_equal(before, after)


def test_dt_is_invariant_to_probe_order():
    """Only the probe ORDER differs between the two runs -- the agent,
    its trial history and the probe set are identical."""
    probes = [np.random.default_rng(s).uniform(0, 1, size=10) for s in range(30)]
    trial_input = np.linspace(0.1, 0.9, 10)

    def run(order):
        agent = ConnectomeInspiredAgent(ConnectomeConfig(), seed=1)
        tracker = TemporalDistanceTracker([probes[i] for i in order])
        tracker.step(agent)
        probs = agent.probabilities(trial_input)
        agent.update(trial_input, 0, 1.0, probs)
        return tracker.step(agent)

    forward = run(list(range(30)))
    backward = run(list(reversed(range(30))))
    assert np.isclose(forward, backward), (forward, backward)


def test_evaluate_is_side_effect_free_for_both_agents():
    from self_fly.agent.factory import make_agent
    from self_fly.config.schema import LearnerConfig

    probes = [np.linspace(0, 1, 10) for _ in range(15)]
    for agent_type in ("baseline", "connectome"):
        agent = make_agent(agent_type, LearnerConfig(), ConnectomeConfig(), seed=3)
        agent.probabilities(np.linspace(0, 1, 10))
        before = agent.full_state()
        agent.evaluate(probes, conditioning_state=agent.recurrent_state())
        assert agent.full_state() == before, agent_type


def test_engine_trial_has_policy_js_delta_from_second_trial_onward():
    engine = ExperimentEngine(default_config(seed=0))
    trial0 = engine.step()
    assert trial0.policy_js_delta is None

    trial1 = engine.step()
    assert trial1.policy_js_delta is not None
    assert trial1.policy_js_delta >= 0.0


def test_connectome_engine_run_keeps_dt_finite():
    config = replace(condition_self_config(seed=0), agent_type="connectome")
    trials = ExperimentEngine(config).run(300)
    deltas = [t.policy_js_delta for t in trials if t.policy_js_delta is not None]
    assert len(deltas) == 299
    assert all(np.isfinite(d) and d >= 0.0 for d in deltas)


def test_classify_dt_series_handles_empty_input():
    assert classify_dt_series([]) == {"n": 0}


def test_classify_dt_series_reports_expected_keys_and_flags_a_spike():
    series = [0.05] * 40 + [0.95] + [0.05] * 40
    result = classify_dt_series(series, window=10)

    assert result["n"] == 81
    assert result["abrupt_change_count"] >= 1
    assert "window_means" in result
    assert "drift_slope" in result
    assert "tail_cv" in result
