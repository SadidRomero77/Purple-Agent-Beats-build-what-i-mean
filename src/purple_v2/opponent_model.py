from __future__ import annotations

from dataclasses import dataclass

from .types import BeliefDistribution, OpponentType


@dataclass
class OpponentModel:
    """Proyeccion simplificada del estilo de respuesta del rival."""

    def projected_risk(self, beliefs: BeliefDistribution) -> float:
        return (
            0.1 * beliefs.get(OpponentType.COOPERATIVE, 0.0)
            + 0.45 * beliefs.get(OpponentType.ADAPTIVE, 0.0)
            + 0.9 * beliefs.get(OpponentType.ADVERSARIAL, 0.0)
            + 0.6 * beliefs.get(OpponentType.NOISY, 0.0)
        )

    def projected_information_gain_from_probe(self, beliefs: BeliefDistribution) -> float:
        return (
            0.2 * beliefs.get(OpponentType.COOPERATIVE, 0.0)
            + 0.55 * beliefs.get(OpponentType.ADAPTIVE, 0.0)
            + 0.35 * beliefs.get(OpponentType.ADVERSARIAL, 0.0)
            + 0.5 * beliefs.get(OpponentType.NOISY, 0.0)
        )

    def likely_style(self, beliefs: BeliefDistribution) -> str:
        top = max(beliefs, key=beliefs.get)
        mapping = {
            OpponentType.COOPERATIVE: "colaborativo",
            OpponentType.ADAPTIVE: "adaptativo",
            OpponentType.ADVERSARIAL: "agresivo-estrategico",
            OpponentType.NOISY: "ruidoso-inconsistente",
        }
        return mapping[top]
