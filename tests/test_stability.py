import numpy as np

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


def test_tv_distance_basic_properties():
    p = {"REAL": 0.9, "FALSA": 0.05, "NO_SE": 0.05}
    assert tv_distance(p, p) == 0.0
    q = {"REAL": 0.1, "FALSA": 0.85, "NO_SE": 0.05}
    assert tv_distance(p, q) > 0.5


def test_stability_detector_ignores_transition_and_fires_on_plateau():
    config = StabilityConfig(
        window_size=50,
        tv_threshold=0.15,
        reward_delta_threshold=0.3,
        cv_threshold=0.5,
        consistency_threshold=0.5,
        consecutive_windows_required=2,
    )
    detector = StabilityDetector(config)
    rng = np.random.default_rng(0)
    idx = 0

    def feed(n: int, probs: list[float]) -> list:
        nonlocal idx
        events = []
        for _ in range(n):
            action = rng.choice(["REAL", "FALSA", "NO_SE"], p=probs)
            reward = 1.0 if action == "REAL" else -1.0
            event = detector.update(_make_trial(idx, action, reward))
            idx += 1
            if event is not None:
                events.append(event)
        return events

    # Two stationary windows: only one comparison so far, not enough to declare stability.
    assert feed(100, [0.9, 0.05, 0.05]) == []

    # A third stationary window makes it two consecutive stable comparisons -> fires.
    events = feed(50, [0.9, 0.05, 0.05])
    assert len(events) == 1

    # Abrupt change in behaviour: the very next comparison must not look stable.
    assert feed(50, [0.1, 0.85, 0.05]) == []

    # Sustained new plateau: eventually fires again once enough consecutive
    # windows (and the reward-baseline CV) reflect the new, stable behaviour.
    # Exactly how many windows that takes depends on how far the stale
    # pre-transition reward mean has to age out of the CV check, so we poll
    # for it rather than hardcoding a window count.
    events: list = []
    windows_fed = 0
    while not events and windows_fed < 6:
        events = feed(50, [0.1, 0.85, 0.05])
        windows_fed += 1
    assert len(events) == 1
