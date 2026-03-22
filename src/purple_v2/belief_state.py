from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig
from .types import BeliefDistribution, Observation, OpponentType
from .utility import clamp, normalize_probs


@dataclass
class BeliefState:
    config: AgentConfig

    def __post_init__(self) -> None:
        self.beliefs: BeliefDistribution = normalize_probs(self.config.priors())

    def likelihood(self, opponent: OpponentType, obs: Observation) -> float:
        c = clamp(obs.compliance)
        a = clamp(obs.aggression)
        d = clamp(obs.deception_signal)
        f = clamp(obs.format_stability)
        m = clamp(obs.ambiguity)

        if opponent == OpponentType.COOPERATIVE:
            return clamp(0.5 * c + 0.2 * f + 0.2 * (1 - a) + 0.1 * (1 - d), 1e-6, 1.0)
        if opponent == OpponentType.ADVERSARIAL:
            return clamp(0.45 * a + 0.35 * d + 0.1 * (1 - c) + 0.1 * (1 - f), 1e-6, 1.0)
        if opponent == OpponentType.NOISY:
            return clamp(0.45 * m + 0.35 * (1 - f) + 0.2 * abs(c - 0.5), 1e-6, 1.0)

        # ADAPTIVE
        adaptive_profile = 1.0 - abs(c - 0.55) - 0.4 * abs(a - 0.45) - 0.3 * abs(d - 0.45)
        return clamp(adaptive_profile, 1e-6, 1.0)

    def update(self, obs: Observation) -> BeliefDistribution:
        posterior = {}
        for opponent, prior in self.beliefs.items():
            post_value = prior * self.likelihood(opponent, obs)
            posterior[opponent] = max(post_value, self.config.likelihood_eps)

        self.beliefs = normalize_probs(posterior, eps=self.config.likelihood_eps)
        return self.beliefs

    def top_hypothesis(self) -> OpponentType:
        return max(self.beliefs, key=self.beliefs.get)
