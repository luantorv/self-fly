from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACTION_LABELS = ("REAL", "FALSA", "NO_SE")


def load_trials_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_action_probabilities(trials: list[dict], output_path: str | Path) -> None:
    probs = np.array([t["action_probs"] for t in trials])
    trial_indices = [t["trial_index"] for t in trials]

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, label in enumerate(ACTION_LABELS):
        ax.plot(trial_indices, probs[:, i], label=f"P({label})")
    ax.set_xlabel("trial")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Action probabilities over time")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_reward(trials: list[dict], output_path: str | Path, window: int = 50) -> None:
    rewards = np.array([t["reward"] for t in trials])
    trial_indices = [t["trial_index"] for t in trials]
    cumulative = np.cumsum(rewards)

    fig, (ax_cum, ax_roll) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_cum.plot(trial_indices, cumulative)
    ax_cum.set_ylabel("cumulative reward")

    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax_roll.plot(trial_indices[window - 1 :], rolling)
    ax_roll.set_ylabel(f"reward (rolling mean, w={window})")
    ax_roll.set_xlabel("trial")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_plots(trials_jsonl_path: str | Path, output_dir: str | Path) -> None:
    trials = load_trials_jsonl(trials_jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_action_probabilities(trials, output_dir / "action_probabilities.png")
    plot_reward(trials, output_dir / "reward.png")
