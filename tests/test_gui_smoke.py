import tkinter as tk

import pytest

from self_fly.config.defaults import default_config
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
    finally:
        root.destroy()
