from __future__ import annotations

from dataclasses import dataclass

from .types import MentalState


@dataclass
class MetaReasoner:
    """Chequeos metacognitivos simples antes de emitir salida."""

    def audit(self, mental_state: MentalState, epistemic_tag: str) -> list[str]:
        findings: list[str] = []

        if "incertidumbre_alta" in mental_state.assumptions and epistemic_tag == "inferencia_fuerte":
            findings.append("inconsistencia: alta incertidumbre con confianza excesiva")

        if "rival_podria_explotar_patrones" in mental_state.assumptions:
            findings.append("accion: aumentar aleatorizacion controlada")

        if mental_state.beliefs.get("context_uncertainty", 0.0) > 0.75:
            findings.append("accion: preferir accion robusta y/o pedir aclaracion")

        if not findings:
            findings.append("ok: sin contradicciones criticas")

        return findings
