from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from self_fly.experiment.stages import StageEvent
from self_fly.experiment.types import PolicySnapshot, Trial


class TrialLogger:
    """Append-only JSONL logging for one run: trials, policy snapshots and
    stage/stability events, each in their own file."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.trials_path = self.run_dir / "trials.jsonl"
        self.snapshots_path = self.run_dir / "policy_snapshots.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        for path in (self.trials_path, self.snapshots_path, self.events_path):
            path.touch(exist_ok=True)

    def log_trial(self, trial: Trial) -> None:
        self._append(self.trials_path, asdict(trial))

    def log_snapshot(self, snapshot: PolicySnapshot) -> None:
        self._append(self.snapshots_path, asdict(snapshot))

    def log_event(self, event: StageEvent) -> None:
        self._append(
            self.events_path,
            {"kind": event.kind, "trial_index": event.trial_index, "payload": event.payload},
        )

    @staticmethod
    def _append(path: Path, data: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
