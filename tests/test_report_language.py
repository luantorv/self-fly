import argparse
from dataclasses import replace

import run_experiment
from self_fly.analysis.report import generate_report
from self_fly.config.defaults import default_config
from self_fly.config.loader import save_config
from stability_helpers import fast_stability_config

FORBIDDEN_PHRASES = (
    "el agente sabe",
    "el agente reconoce",
    "el agente es consciente",
    "el agente comprende",
    "el agente percibe que",
    "agente autoconsciente",
    "cree que es real",
    "duda de si misma",
)


def _fast_config(seed: int = 0):
    config = default_config(seed=seed)
    fast_stability = fast_stability_config()
    stage0, stage1 = config.stages
    stage1 = replace(stage1, max_trials=200)
    return replace(config, stability=fast_stability, stages=[stage0, stage1])


def test_report_has_both_sections_and_no_anthropomorphic_claims(tmp_path):
    config_path = tmp_path / "config.json"
    save_config(_fast_config(), config_path)
    args = argparse.Namespace(
        config=config_path,
        seed=None,
        seeds=None,
        agent=None,
        condition=None,
        n_trials=500,
        run_name=None,
        runs_dir=tmp_path / "runs",
    )
    summary = run_experiment.run(args)

    report = generate_report(summary["run_dir"])

    assert "## Observaciones" in report
    assert "## Interpretacion" in report

    observations_section = report.split("## Interpretacion")[0]
    interpretation_section = report.split("## Interpretacion")[1]
    assert observations_section.strip()
    assert interpretation_section.strip()

    lowered = report.lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in lowered, f"anthropomorphic claim found: {phrase!r}"
