from __future__ import annotations

from dataclasses import dataclass

from .belief_state import BeliefState
from .config import AgentConfig
from .meta_reasoner import MetaReasoner
from .mind_model import MindModel
from .mixed_policy import MixedStrategyPolicy
from .opponent_model import OpponentModel
from .output_validator import OutputValidator
from .planner import UtilityPlanner
from .types import ActionType, AgentContext, Decision, Observation
from .utility import clamp
from .voi_gate import VOIGate


@dataclass
class PurpleAgentV2:
    config: AgentConfig

    def __post_init__(self) -> None:
        self.belief_state = BeliefState(self.config)
        self.opponent_model = OpponentModel()
        self.voi_gate = VOIGate(self.config)
        self.planner = UtilityPlanner(self.config.utility_weights)
        self.policy = MixedStrategyPolicy()
        self.mind_model = MindModel()
        self.meta_reasoner = MetaReasoner()
        self.validator = OutputValidator()

    def _temperature(self, context: AgentContext, projected_risk: float) -> float:
        dynamic = self.config.base_temperature + 0.2 * clamp(context.uncertainty) + 0.25 * clamp(projected_risk)
        return clamp(dynamic, self.config.min_temperature, self.config.max_temperature)

    def decide(self, obs: Observation, context: AgentContext) -> tuple[Decision, dict]:
        beliefs = self.belief_state.update(obs)

        projected_risk = self.opponent_model.projected_risk(beliefs)
        projected_probe_gain = self.opponent_model.projected_information_gain_from_probe(beliefs)
        rival_style = self.opponent_model.likely_style(beliefs)

        ask_flag = self.voi_gate.should_ask(obs, context.uncertainty)
        utilities = self.planner.evaluate(
            context=context,
            projected_risk=projected_risk,
            projected_probe_gain=projected_probe_gain,
            should_ask=ask_flag,
        )

        temperature = self._temperature(context, projected_risk)
        action, dist = self.policy.choose(utilities, temperature)
        confidence = dist[action]

        mental_state = self.mind_model.build(beliefs, context, rival_style)
        epistemic_status = self.mind_model.epistemic_tag(confidence)
        findings = self.meta_reasoner.audit(mental_state, epistemic_status)

        decision = Decision(
            action=action,
            confidence=confidence,
            expected_utility=utilities[action],
            rationale=[
                f"rival_style={rival_style}",
                f"projected_risk={projected_risk:.3f}",
                f"ask_flag={ask_flag}",
                *findings,
            ],
            ask_for_clarification=ask_flag,
        )

        debug = {
            "beliefs": {k.value: v for k, v in beliefs.items()},
            "policy_distribution": {k.value: v for k, v in dist.items()},
            "temperature": temperature,
            "epistemic_status": epistemic_status,
            "exploitability_index": self.policy.exploitability_index(dist),
        }
        return decision, debug

    def build_payload(self, prompt: str, decision: Decision, epistemic_status: str) -> dict:
        templates = {
            ActionType.ANSWER_DIRECT: f"Respuesta directa propuesta: {prompt[:120]}",
            ActionType.ASK_CLARIFICATION: "Necesito una aclaracion puntual para maximizar EV.",
            ActionType.PROBE_OPPONENT: "Haré una pregunta de sondeo para estimar mejor tu intencion.",
            ActionType.HEDGE_SAFETY: "Respuesta robusta bajo incertidumbre, con supuestos explicitos.",
            ActionType.SAFE_FALLBACK: "Fallback seguro: salida conservadora para evitar error critico.",
        }

        payload = {
            "action": decision.action.value,
            "answer": templates[decision.action],
            "confidence": round(decision.confidence, 4),
            "epistemic_status": epistemic_status,
            "ask_for_clarification": decision.ask_for_clarification,
            "rationale": decision.rationale,
            "expected_utility": round(decision.expected_utility, 4),
        }

        errors = self.validator.validate(payload)
        if errors:
            return self._safe_fallback_payload(errors)
        return payload

    @staticmethod
    def _safe_fallback_payload(errors: list[str]) -> dict:
        return {
            "action": ActionType.SAFE_FALLBACK.value,
            "answer": "Fallback activado por validacion de salida.",
            "confidence": 0.4,
            "epistemic_status": "conjetura",
            "ask_for_clarification": True,
            "rationale": ["output_validator_error", *errors],
        }
