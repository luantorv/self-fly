import inspect
import tkinter as tk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from self_fly.stimuli.features import FeatureVector  # noqa: E402
from self_fly.visualization.renderers.matplotlib_renderer import MatplotlibRenderer  # noqa: E402
from self_fly.visualization.renderers.renderer_3d import Renderer3D  # noqa: E402
from self_fly.visualization.renderers.tk_renderer import TkRenderer  # noqa: E402

_SAMPLE_FEATURES = FeatureVector(
    shape_roundness=0.6,
    color_hue=0.3,
    color_saturation=0.7,
    orientation_sin=0.5,
    orientation_cos=0.5,
    texture_density=0.4,
    movement_speed=0.3,
    movement_direction=0.2,
    aspect_ratio=0.5,
    context_level=0.6,
)

_FORBIDDEN_PARAM_NAMES = {"label", "stage", "category_id", "ground_truth_label"}


def _make_tk_root():
    try:
        return tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display available for tkinter: {exc}")


def test_tk_renderer_draws_without_exception():
    root = _make_tk_root()
    try:
        canvas = tk.Canvas(root)
        TkRenderer(canvas).render(_SAMPLE_FEATURES)
    finally:
        root.destroy()


def test_matplotlib_renderer_draws_without_exception():
    fig, ax = plt.subplots()
    try:
        MatplotlibRenderer(ax).render(_SAMPLE_FEATURES)
    finally:
        plt.close(fig)


def test_renderer_3d_is_an_explicit_placeholder():
    with pytest.raises(NotImplementedError):
        Renderer3D().render(_SAMPLE_FEATURES)


@pytest.mark.parametrize("cls", [TkRenderer, MatplotlibRenderer, Renderer3D])
def test_renderer_signatures_never_accept_privileged_parameters(cls):
    """renderers/*.py isn't covered by test_no_privileged_leak.py's AST
    check (only render.py and agent/**/*.py are) -- this is the safety net
    for that gap: no renderer's render() may accept label/stage/
    category_id/ground_truth_label, mirroring the same boundary."""
    signature = inspect.signature(cls.render)
    assert not (set(signature.parameters) & _FORBIDDEN_PARAM_NAMES)
