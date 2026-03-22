from __future__ import annotations

from dataclasses import dataclass

from .types import ActionType, ActionUtilities, AgentContext, UtilityWeights
from .utility import clamp


@dataclass
class UtilityPlanner:
    weights: UtilityWeights

    def evaluate(
        self,
        context: AgentContext,
        projected_risk: float,
        projected_probe_gain: float,
        should_ask: bool,
    ) -> ActionUtilities:
        uncertainty = clamp(context.uncertainty)
        risk = clamp(projected_risk)
        probe_gain = clamp(projected_probe_gain)

        direct = 0.9 - 0.5 * uncertainty - 0.4 * risk
        ask = 0.45 + (0.35 if should_ask else -0.15) - 0.1 * risk
        probe = 0.35 + 0.5 * probe_gain - 0.2 * uncertainty
        hedge = 0.5 + 0.3 * risk + 0.2 * uncertainty
        safe = 0.3 + 0.4 * risk + 0.35 * uncertainty

        # Ajustes por costo global de riesgo/tiempo/formato.
        direct -= self.weights.risk * risk
        probe -= self.weights.risk * (risk * 0.6)
        hedge -= self.weights.latency * 0.15
        ask -= self.weights.latency * 0.1
        safe -= self.weights.exploration * 0.2

        return {
            ActionType.ANSWER_DIRECT: direct,
            ActionType.ASK_CLARIFICATION: ask,
            ActionType.PROBE_OPPONENT: probe,
            ActionType.HEDGE_SAFETY: hedge,
            ActionType.SAFE_FALLBACK: safe,
        }
