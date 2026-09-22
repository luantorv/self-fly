from self_fly.config.schema import StabilityConfig
from self_fly.experiment.stability import StabilityDetector, tv_distance
from self_fly.experiment.types import Trial


def _make_trial(idx: int, action: str, reward: float, category: str = "c0") -> Trial:
    return Trial(
        trial_index=idx,
        stage=0,
        stimulus_id=f"s{idx}",
        category_id=category,
        features=tuple([0.0] * 10),
        action=action,
        action_probs=(0.33, 0.33, 0.34),
        reward=reward,
        ground_truth_label="real",
        policy_weight_norm=0.0,
        timestamp="t",
    )


def _config(**kwargs) -> StabilityConfig:
    base = dict(
        window_size=10,
        min_accuracy=0.70,
        max_median_policy_delta=0.002,
        consecutive_windows_required=2,
    )
    base.update(kwargs)
    return StabilityConfig(**base)


def _feed(detector, n, delta, start=0):
    events = []
    for i in range(n):
        event = detector.update(_make_trial(start + i, "REAL", 1.0), delta)
        if event is not None:
            events.append(event)
    return events


def test_tv_distance_basic_properties():
    p = {"REAL": 0.9, "FALSA": 0.05, "NO_SE": 0.05}
    assert tv_distance(p, p) == 0.0
    q = {"REAL": 0.1, "FALSA": 0.85, "NO_SE": 0.05}
    assert tv_distance(p, q) > 0.5


def test_fires_only_after_enough_consecutive_settled_windows():
    detector = StabilityDetector(_config(), accuracy_fn=lambda: 0.8)

    assert _feed(detector, 10, 0.0001) == []  # one window: not enough yet
    events = _feed(detector, 10, 0.0001, start=10)  # second consecutive window
    assert len(events) == 1
    assert events[0].accuracy == 0.8
    assert events[0].median_policy_delta <= 0.002


def test_drifting_policy_never_fires():
    detector = StabilityDetector(_config(), accuracy_fn=lambda: 0.9)
    assert _feed(detector, 100, 0.5) == []
    assert detector.max_consecutive_stable_observed == 0


def test_incompetent_policy_never_fires_even_when_perfectly_settled():
    detector = StabilityDetector(_config(), accuracy_fn=lambda: 0.55)
    assert _feed(detector, 100, 0.0) == []


def test_streak_resets_on_a_single_bad_window():
    detector = StabilityDetector(_config(consecutive_windows_required=3), accuracy_fn=lambda: 0.8)
    _feed(detector, 20, 0.0001)  # streak of 2
    assert detector.max_consecutive_stable_observed == 2
    _feed(detector, 10, 0.5, start=20)  # one drifting window breaks it
    assert _feed(detector, 20, 0.0001, start=30) == []  # only back to 2


def test_criterion_is_unaffected_by_zero_reward_stimuli():
    """The previous coefficient-of-variation criterion divided by the mean
    reward, so mixing in zero-reward stimuli made stability strictly harder
    to reach even with identical behaviour. The criterion must no longer
    depend on reward magnitude at all."""
    settled = 0.0001

    rewarded = StabilityDetector(_config(), accuracy_fn=lambda: 0.8)
    for i in range(20):
        rewarded.update(_make_trial(i, "REAL", 1.0), settled)

    zero_reward = StabilityDetector(_config(), accuracy_fn=lambda: 0.8)
    for i in range(20):
        zero_reward.update(_make_trial(i, "REAL", 0.0), settled)

    assert (
        rewarded.max_consecutive_stable_observed
        == zero_reward.max_consecutive_stable_observed
        == 2
    )


def test_reset_clears_history_at_a_stage_boundary():
    detector = StabilityDetector(_config(), accuracy_fn=lambda: 0.8)
    _feed(detector, 10, 0.0001)
    detector.reset()
    assert detector.max_consecutive_stable_observed == 0
    assert _feed(detector, 10, 0.0001, start=10) == []  # streak restarted
