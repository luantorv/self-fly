from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_run_summary(run_dir: Path) -> dict:
    """Purely descriptive, post-hoc read of a single runs/<id>/ directory --
    never re-runs anything, never infers a value that isn't already logged."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics_summary = json.loads((run_dir / "metrics_summary.json").read_text())
    events = _read_jsonl(run_dir / "events.jsonl")

    snapshot_trial_by_name = {
        event["payload"]["name"]: event["trial_index"]
        for event in events
        if event["kind"] == "capture_snapshot"
    }
    stage_changed_events = [e for e in events if e["kind"] == "stage_changed"]

    # Block 3 introduces pi_2_unstable / pi_2_timeout alongside bare pi_2;
    # any of them counts as "stage 1 concluded", but the exact name matters
    # for telling a real convergence apart from a forced stop.
    pi2_name = next((name for name in snapshot_trial_by_name if name.startswith("pi_2")), None)

    return {
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "agent_type": manifest.get("agent_type"),
        "experimental_condition": manifest.get("experimental_condition"),
        "status": manifest["status"],
        "stage0_converged": "pi_0" in snapshot_trial_by_name,
        "pi0_trial": snapshot_trial_by_name.get("pi_0"),
        "self_stage_entered": len(stage_changed_events) > 0,
        "pi1_reached": "pi_1" in snapshot_trial_by_name,
        "pi1_trial": snapshot_trial_by_name.get("pi_1"),
        "pi2_reached": pi2_name is not None,
        "pi2_snapshot_name": pi2_name,
        "pi2_trial": snapshot_trial_by_name.get(pi2_name) if pi2_name else None,
        "trial_count": metrics_summary.get("trial_count"),
        "mean_reward": metrics_summary.get("mean_reward"),
        "mean_entropy": metrics_summary.get("mean_entropy"),
        "entropy_at_pi0": metrics_summary.get("entropy_at_pi0"),
        "entropy_at_pi1": metrics_summary.get("entropy_at_pi1"),
        "entropy_at_pi2": metrics_summary.get("entropy_at_pi2"),
        "action_frequencies": metrics_summary.get("action_frequencies"),
    }


def _numeric_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": variance**0.5, "min": min(values), "max": max(values), "n": len(values)}


_NUMERIC_SUMMARY_FIELDS = ["mean_reward", "mean_entropy", "pi0_trial", "pi1_trial", "pi2_trial"]


def aggregate_runs(run_dirs: list[Path]) -> dict:
    """Descriptive multi-seed aggregation: per-seed summaries plus
    group-level mean/std/min/max over numeric fields. No pass/fail verdict:
    a seed that never reached pi_2 still contributes its entry (pi2_trial
    stays None) instead of being silently dropped from the average."""
    summaries = [load_run_summary(d) for d in run_dirs]

    aggregated = {
        field_name: _numeric_stats([s[field_name] for s in summaries if s.get(field_name) is not None])
        for field_name in _NUMERIC_SUMMARY_FIELDS
    }

    return {
        "n_seeds": len(summaries),
        "stage0_convergence_rate": (
            sum(1 for s in summaries if s["stage0_converged"]) / len(summaries) if summaries else None
        ),
        "pi2_reached_rate": (
            sum(1 for s in summaries if s["pi2_reached"]) / len(summaries) if summaries else None
        ),
        "per_seed": summaries,
        "aggregated": aggregated,
    }


def compare_conditions(runs_by_group: dict[str, list[Path]]) -> dict:
    """Descriptive comparison across experimental conditions -- or agent
    types, or any other grouping the caller's dict keys represent -- built
    entirely from per-group aggregate_runs() calls. Deliberately offers no
    ranking, "best", or "winner" field: differences are reported, never
    declared a verdict."""
    return {group: aggregate_runs(run_dirs) for group, run_dirs in runs_by_group.items()}
