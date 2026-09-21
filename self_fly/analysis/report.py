from __future__ import annotations

from pathlib import Path

from .aggregate import _read_jsonl, load_run_summary

_INTERPRETATION_DISCLAIMER = (
    "Estas observaciones describen comportamiento medible de la politica "
    "(distribucion de acciones, entropia, distancias de Jensen-Shannon, "
    "recompensa). No constituyen evidencia de conciencia, autoconciencia, "
    "comprension semantica ni reconocimiento de si mismo por parte del "
    "agente. REAL, FALSA y NO_SE son acciones sin semantica especial para "
    "el modelo; SELF_A/SELF_B/OTHER/CONTROL son categorias de estimulo, no "
    "afirmaciones sobre lo que el agente 'sabe' o 'percibe'."
)

_SELF_MULTI_DISCLAIMER = (
    "Esta corrida incluye las categorias SELF_A/SELF_B/OTHER. Cualquier "
    "afirmacion de similitud o disimilitud conductual entre ellas debe "
    "basarse en las distancias de Jensen-Shannon calculadas explicitamente "
    "(self_fly.experiment.policy_distance.js_distance) sobre las "
    "probabilidades de accion evaluadas en cada estimulo congelado, y debe "
    "formularse como 'la politica trata a X e Y de forma mas/menos similar "
    "que a Z' -- nunca como reconocimiento, identidad, o autoconciencia."
)


def generate_report(run_dir: str | Path) -> str:
    """Markdown report for a single run, split into two sections that must
    never blur together: Observaciones (numbers, read straight from the
    run's own JSONL/JSON files, nothing inferred) and Interpretacion
    (fixed disclaimer text, never free-form language)."""
    run_dir = Path(run_dir)
    summary = load_run_summary(run_dir)
    events = _read_jsonl(run_dir / "events.jsonl")

    stage_outcomes = [e for e in events if e["kind"] == "stage_outcome"]
    auxiliary_events = [e for e in events if e["kind"] == "auxiliary_agent_outcome"]

    lines = [f"# Informe de corrida: {summary['run_id']}", "", "## Observaciones", ""]
    lines += [
        f"- Agente: {summary['agent_type']}",
        f"- Condicion experimental: {summary['experimental_condition']}",
        f"- Seed: {summary['seed']}",
        f"- Estado final de la corrida: {summary['status']}",
        f"- Etapa 0 convergio: {summary['stage0_converged']} (trial {summary['pi0_trial']})",
        f"- Etapa autorreferencial iniciada: {summary['self_stage_entered']}",
        f"- pi_1 alcanzado: {summary['pi1_reached']} (trial {summary['pi1_trial']})",
        f"- pi_2 (o variante) alcanzado: {summary['pi2_reached']} -- "
        f"snapshot: {summary['pi2_snapshot_name']} (trial {summary['pi2_trial']})",
        f"- Recompensa media: {summary['mean_reward']}",
        f"- Entropia media: {summary['mean_entropy']}",
        f"- Entropia en pi_0 / pi_1 / pi_2: "
        f"{summary['entropy_at_pi0']} / {summary['entropy_at_pi1']} / {summary['entropy_at_pi2']}",
        f"- Frecuencia de acciones: {summary['action_frequencies']}",
    ]

    lines += ["", "### Desenlaces por etapa (StageOutcome)"]
    if stage_outcomes:
        for event in stage_outcomes:
            payload = event["payload"]
            lines.append(
                f"- Etapa {payload.get('stage')}: outcome={payload.get('outcome')}, "
                f"snapshot={payload.get('snapshot_name')}, "
                f"trials_en_etapa={payload.get('trials_in_stage')}, "
                f"entropia={payload.get('entropy')}, "
                f"JS_a_inicio_de_etapa={payload.get('js_to_stage_entry')}"
            )
    else:
        lines.append("- (ninguna etapa concluyo durante esta corrida)")

    if auxiliary_events:
        lines += ["", "### Agente auxiliar (OTHER)"]
        for event in auxiliary_events:
            lines.append(f"- outcome={event['payload'].get('outcome')}")

    lines += ["", "## Interpretacion", "", _INTERPRETATION_DISCLAIMER]
    if summary["experimental_condition"] == "self_multi":
        lines += ["", _SELF_MULTI_DISCLAIMER]

    return "\n".join(lines)
