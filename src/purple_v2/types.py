from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class OpponentType(str, Enum):
    COOPERATIVE = "cooperative"
    ADAPTIVE = "adaptive"
    ADVERSARIAL = "adversarial"
    NOISY = "noisy"


class ActionType(str, Enum):
    ANSWER_DIRECT = "answer_direct"
    ASK_CLARIFICATION = "ask_clarification"
    PROBE_OPPONENT = "probe_opponent"
    HEDGE_SAFETY = "hedge_safety"
    SAFE_FALLBACK = "safe_fallback"


@dataclass
class Observation:
    """Senales observadas por ronda. Todas se esperan en [0, 1]."""

    ambiguity: float
    compliance: float
    aggression: float
    deception_signal: float
    format_stability: float


@dataclass
class AgentContext:
    round_index: int
    uncertainty: float
    response_time_budget_ms: int = 2500


@dataclass
class UtilityWeights:
    risk: float = 0.35
    latency: float = 0.15
    format_error: float = 0.4
    exploration: float = 0.1


@dataclass
class MentalState:
    beliefs: Dict[str, float]
    desires: Dict[str, float]
    intentions: List[str]
    assumptions: List[str] = field(default_factory=list)


@dataclass
class Decision:
    action: ActionType
    confidence: float
    expected_utility: float
    rationale: List[str]
    ask_for_clarification: bool


BeliefDistribution = Dict[OpponentType, float]
ActionUtilities = Dict[ActionType, float]
