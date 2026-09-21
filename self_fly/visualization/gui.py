from __future__ import annotations

import tkinter as tk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from self_fly.config.schema import ExperimentConfig  # noqa: E402
from self_fly.experiment.engine import ExperimentEngine  # noqa: E402
from self_fly.experiment.types import Trial  # noqa: E402
from self_fly.io.run_manager import RunManager  # noqa: E402
from self_fly.metrics.collectors import MetricsCollector  # noqa: E402
from self_fly.stimuli.features import FeatureVector  # noqa: E402

from .render import render_stimulus  # noqa: E402

ACTION_ORDER = ("REAL", "FALSA", "NO_SE")

STAGE_LABELS = {
    0: "ETAPA 0 · aprendizaje normal",
    1: "ETAPA 1 · periodo de estimulo autorreferencial activo",
    2: "ETAPA 2 · conflicto leve",
    3: "ETAPA 3 · conflicto fuerte",
    4: "ETAPA 4 · conflicto de tres vias (SELF_A/SELF_B/OTHER)",
}

# Observer-facing only (see README's agent/visualization separation): the
# GUI is allowed to know which frozen stimulus category is on screen,
# because this drives a banner shown to the human, never anything fed
# back into agent/. NO_SPECIAL_LABELS marks the ordinary REAL/FALSA case.
SPECIAL_STIMULUS_BANNERS = {
    "self_a": "estimulo AUTORREFERENCIAL (SELF_A) en pantalla",
    "self_b": "estimulo AUTORREFERENCIAL variante (SELF_B) en pantalla",
    "other": "estimulo de OTRO agente en pantalla",
    "control": "estimulo de CONTROL (congelado, no autorreferencial) en pantalla",
}

HISTORY_LIMIT = 800


class SelfFlyApp:
    """Drives ExperimentEngine.step() from tkinter's event loop. The engine
    itself is identical to the one run_experiment.py uses headless -- this
    class only observes and renders it."""

    def __init__(
        self,
        root: tk.Tk,
        config: ExperimentConfig,
        run_manager: RunManager | None = None,
        step_delay_ms: int = 30,
    ):
        self.root = root
        self.root.title("SELF-FLY")
        self.step_delay_ms = step_delay_ms
        self.run_manager = run_manager

        self.metrics = MetricsCollector()
        logger = run_manager.logger if run_manager is not None else None
        self.engine = ExperimentEngine(config, logger=logger, metrics=self.metrics)

        self.history_probs: dict[str, list[float]] = {name: [] for name in ACTION_ORDER}
        self.history_reward: list[float] = []

        self._build_widgets()
        self._schedule_step()

    def _build_widgets(self) -> None:
        self.stage_banner = tk.Label(self.root, text="", font=("Sans", 14, "bold"), pady=8)
        self.stage_banner.pack(fill="x")

        self.special_stimulus_banner = tk.Label(
            self.root, text="", font=("Sans", 10, "italic"), pady=2, fg="#a33"
        )
        self.special_stimulus_banner.pack(fill="x")

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main)
        left.pack(side="left", padx=10, pady=10)

        self.canvas = tk.Canvas(
            left, width=220, height=220, bg="white", highlightthickness=1, highlightbackground="#999"
        )
        self.canvas.pack()

        buttons_frame = tk.Frame(left)
        buttons_frame.pack(pady=8)
        tk.Label(buttons_frame, text="Decisión automática del agente", font=("Sans", 9, "italic")).pack()
        button_row = tk.Frame(buttons_frame)
        button_row.pack()
        self.action_buttons: dict[str, tk.Button] = {}
        for action_name in ACTION_ORDER:
            btn = tk.Button(button_row, text=action_name, state="disabled", width=8, relief="raised")
            btn.pack(side="left", padx=4)
            self.action_buttons[action_name] = btn
        self._default_button_bg = next(iter(self.action_buttons.values())).cget("bg")

        right = tk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.info_label = tk.Label(right, justify="left", anchor="w", font=("Courier", 10))
        self.info_label.pack(fill="x")

        self.prob_bars_canvas = tk.Canvas(
            right, width=320, height=80, bg="white", highlightthickness=1, highlightbackground="#999"
        )
        self.prob_bars_canvas.pack(pady=8)

        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.ax_probs = self.figure.add_subplot(211)
        self.ax_reward = self.figure.add_subplot(212)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _schedule_step(self) -> None:
        self.root.after(self.step_delay_ms, self._step_and_refresh)

    def _step_and_refresh(self) -> None:
        trial = self.engine.step()
        self._refresh(trial)
        self._schedule_step()

    def _refresh(self, trial: Trial) -> None:
        stage = self.engine.current_stage
        self.stage_banner.config(text=STAGE_LABELS.get(stage, f"ETAPA {stage}"))
        self.special_stimulus_banner.config(
            text=SPECIAL_STIMULUS_BANNERS.get(trial.ground_truth_label, "")
        )

        features = FeatureVector.from_tuple(trial.features)
        render_stimulus(self.canvas, features, cx=110, cy=110, size=70)

        for name, btn in self.action_buttons.items():
            is_taken = name == trial.action
            btn.config(
                relief="sunken" if is_taken else "raised",
                bg="#cfe8ff" if is_taken else self._default_button_bg,
            )

        probs = dict(zip(ACTION_ORDER, trial.action_probs))
        self._draw_prob_bars(probs)

        snapshots = ", ".join(self.engine.policy_snapshots.keys()) or "-"
        status = "finalizado" if self.engine.stage_machine.finished else "en curso"
        entropy = self.metrics.entropy_history[-1] if self.metrics.entropy_history else None
        entropy_text = f"{entropy:.3f}" if entropy is not None else "-"
        dt_text = f"{trial.policy_js_delta:.4f}" if trial.policy_js_delta is not None else "-"
        self.info_label.config(
            text=(
                f"Etapa:           {stage}\n"
                f"Ensayo:          {trial.trial_index}\n"
                f"Recompensa:      {trial.reward:+.2f}\n"
                f"Rec. acumulada:  {self.metrics.cumulative_reward:+.2f}\n"
                f"P(REAL):         {probs['REAL']:.3f}\n"
                f"P(FALSA):        {probs['FALSA']:.3f}\n"
                f"P(NO_SE):        {probs['NO_SE']:.3f}\n"
                f"H(pi):           {entropy_text}\n"
                f"D_t = JS(t,t-1): {dt_text}\n"
                f"Snapshots:       {snapshots}\n"
                f"Estado:          {status}\n"
            )
        )

        for name in ACTION_ORDER:
            history = self.history_probs[name]
            history.append(probs[name])
            del history[:-HISTORY_LIMIT]
        self.history_reward.append(trial.reward)
        del self.history_reward[:-HISTORY_LIMIT]

        if trial.trial_index % 10 == 0:
            self._redraw_figure()

    def _draw_prob_bars(self, probs: dict[str, float]) -> None:
        self.prob_bars_canvas.delete("bar")
        bar_width = 260
        for i, name in enumerate(ACTION_ORDER):
            y = 12 + i * 22
            p = probs[name]
            self.prob_bars_canvas.create_rectangle(
                10, y, 10 + bar_width * p, y + 16, fill="#5b9bd5", tags="bar"
            )
            self.prob_bars_canvas.create_text(
                10 + bar_width + 30, y + 8, text=f"{name} {p:.2f}", anchor="w", tags="bar"
            )

    def _redraw_figure(self) -> None:
        self.ax_probs.clear()
        for name in ACTION_ORDER:
            self.ax_probs.plot(self.history_probs[name], label=name)
        self.ax_probs.set_ylim(0, 1)
        self.ax_probs.legend(fontsize=6, loc="upper left")
        self.ax_probs.set_title("P(acción)", fontsize=8)

        self.ax_reward.clear()
        self.ax_reward.plot(self.history_reward)
        self.ax_reward.set_title("recompensa", fontsize=8)

        self.figure_canvas.draw()
