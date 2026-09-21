from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from self_fly.config.loader import config_to_dict
from self_fly.config.schema import ExperimentConfig

from .logger import TrialLogger


def canonical_config_hash(config: ExperimentConfig) -> str:
    canonical = json.dumps(config_to_dict(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _make_run_id(run_name: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{timestamp}_{run_name}"


class RunManager:
    """Creates runs/<run_id>/, persists config+seed+manifest up front, and
    hands out the TrialLogger every module in a run should share."""

    def __init__(self, config: ExperimentConfig, base_dir: str | Path = "runs"):
        self.config = config
        self.run_id = _make_run_id(config.run_name)
        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.logger = TrialLogger(self.run_dir)

        self._config_hash = canonical_config_hash(config)
        self._started_at = dt.datetime.now(dt.timezone.utc).isoformat()

        (self.run_dir / "config.json").write_text(
            json.dumps(config_to_dict(config), indent=2, ensure_ascii=False)
        )
        self._write_manifest(status="running", finished_at=None)

    def _write_manifest(self, status: str, finished_at: str | None) -> None:
        manifest = {
            "run_id": self.run_id,
            "seed": self.config.seed,
            "config_hash": self._config_hash,
            "started_at": self._started_at,
            "finished_at": finished_at,
            "status": status,
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )

    def write_metrics_summary(self, summary: dict) -> None:
        (self.run_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

    def finalize(self, status: str = "completed") -> None:
        finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self._write_manifest(status=status, finished_at=finished_at)
