from __future__ import annotations

from dataclasses import dataclass, field

from .types import OpponentType, UtilityWeights


@dataclass
class AgentConfig:
    prior_cooperative: float = 0.25
    prior_adaptive: float = 0.4
    prior_adversarial: float = 0.25
    prior_noisy: float = 0.1

    voi_threshold: float = 0.08
    ask_cost: float = 0.12

    base_temperature: float = 0.9
    min_temperature: float = 0.45
    max_temperature: float = 1.2

    utility_weights: UtilityWeights = field(default_factory=UtilityWeights)

    likelihood_eps: float = 1e-6

    def priors(self) -> dict[OpponentType, float]:
        return {
            OpponentType.COOPERATIVE: self.prior_cooperative,
            OpponentType.ADAPTIVE: self.prior_adaptive,
            OpponentType.ADVERSARIAL: self.prior_adversarial,
            OpponentType.NOISY: self.prior_noisy,
        }
