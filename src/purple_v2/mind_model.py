from __future__ import annotations

from dataclasses import dataclass

from .types import AgentContext, BeliefDistribution, MentalState, OpponentType


@dataclass
class MindModel:
    """Modelo BDI simplificado para teoria de la mente."""

    def build(self, beliefs: BeliefDistribution, context: AgentContext, rival_style: str) -> MentalState:
        adversarial_prob = beliefs.get(OpponentType.ADVERSARIAL, 0.0)
        cooperative_prob = beliefs.get(OpponentType.COOPERATIVE, 0.0)

        belief_map = {
            "rival_adversarial_prob": adversarial_prob,
            "rival_cooperative_prob": cooperative_prob,
            "context_uncertainty": context.uncertainty,
        }

        desires = {
            "maximize_score": 1.0,
            "minimize_exploitability": 0.95,
            "preserve_format": 0.98,
            "maintain_latency": 0.7,
        }

        intentions = [
            "adapt_to_{}".format(rival_style),
            "act_with_epistemic_discipline",
        ]

        assumptions = []
        if context.uncertainty > 0.7:
            assumptions.append("incertidumbre_alta")
        if adversarial_prob > 0.6:
            assumptions.append("rival_podria_explotar_patrones")

        return MentalState(
            beliefs=belief_map,
            desires=desires,
            intentions=intentions,
            assumptions=assumptions,
        )

    @staticmethod
    def epistemic_tag(confidence: float) -> str:
        if confidence > 0.8:
            return "inferencia_fuerte"
        if confidence > 0.55:
            return "inferencia"
        return "conjetura"
