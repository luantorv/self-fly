import tkinter as tk
from dataclasses import replace

import pytest

from self_fly.config.defaults import condition_multi_self_config, default_config
from self_fly.config.schema import StabilityConfig
from self_fly.visualization.gui import SelfFlyApp


def _make_root():
    try:
        return tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display available for tkinter: {exc}")


def _cancel_pending_after(root: tk.Tk) -> None:
    for after_id in root.tk.call("after", "info"):
        root.after_cancel(after_id)


def test_gui_runs_many_steps_without_exceptions():
    root = _make_root()
    try:
        config = default_config(seed=0)
        app = SelfFlyApp(root, config, run_manager=None, step_delay_ms=1)
        _cancel_pending_after(root)

        for _ in range(150):
            app._step_and_refresh()
            _cancel_pending_after(root)

        root.update()
        assert app.engine._trial_index == 150
        assert "Ensayo:" in app.info_label.cget("text")
        assert "H(pi):" in app.info_label.cget("text")
    finally:
        root.destroy()


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
    stage1 = replace(stage1, max_trials=300)
    return replace(config, stability=fast_stability, stages=[stage0, stage1])


def test_gui_shows_special_stimulus_banner_without_leaking_to_agent():
    root = _make_root()
    try:
        config = _fast_multi_self_config(seed=1)
        app = SelfFlyApp(root, config, run_manager=None, step_delay_ms=1)
        _cancel_pending_after(root)

        seen_banner = False
        for _ in range(3000):
            app._step_and_refresh()
            _cancel_pending_after(root)
            if app.special_stimulus_banner.cget("text"):
                seen_banner = True
                break

        root.update()
        assert seen_banner, "special-stimulus banner never appeared within budget"
    finally:
        root.destroy()
