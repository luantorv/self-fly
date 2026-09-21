import json

from self_fly.config.defaults import default_config
from self_fly.config.loader import config_from_dict
from self_fly.experiment.engine import ExperimentEngine
from self_fly.io.run_manager import RunManager
from self_fly.metrics.collectors import MetricsCollector


def _read_jsonl(path):
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line]


def _strip_timestamps(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "timestamp"} for r in records]


def test_run_manager_writes_config_manifest_and_logs(tmp_path):
    config = default_config(seed=0, run_name="io_test")
    run_manager = RunManager(config, base_dir=tmp_path)

    engine = ExperimentEngine(config, logger=run_manager.logger, metrics=MetricsCollector())
    engine.run(30)
    run_manager.write_metrics_summary({"trial_count": 30})
    run_manager.finalize()

    assert (run_manager.run_dir / "config.json").exists()
    assert (run_manager.run_dir / "manifest.json").exists()
    assert (run_manager.run_dir / "metrics_summary.json").exists()

    saved_config = config_from_dict(json.loads((run_manager.run_dir / "config.json").read_text()))
    assert saved_config == config

    manifest = json.loads((run_manager.run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 0
    assert manifest["finished_at"] is not None

    trials = _read_jsonl(run_manager.run_dir / "trials.jsonl")
    assert len(trials) == 30


def test_same_seed_produces_identical_trials_ignoring_timestamps(tmp_path):
    config = default_config(seed=42, run_name="determinism_test")

    run_a = RunManager(config, base_dir=tmp_path / "a")
    engine_a = ExperimentEngine(config, logger=run_a.logger)
    engine_a.run(80)

    run_b = RunManager(config, base_dir=tmp_path / "b")
    engine_b = ExperimentEngine(config, logger=run_b.logger)
    engine_b.run(80)

    trials_a = _strip_timestamps(_read_jsonl(run_a.run_dir / "trials.jsonl"))
    trials_b = _strip_timestamps(_read_jsonl(run_b.run_dir / "trials.jsonl"))
    assert trials_a == trials_b

    snapshots_a = _strip_timestamps(_read_jsonl(run_a.run_dir / "policy_snapshots.jsonl"))
    snapshots_b = _strip_timestamps(_read_jsonl(run_b.run_dir / "policy_snapshots.jsonl"))
    assert snapshots_a == snapshots_b
