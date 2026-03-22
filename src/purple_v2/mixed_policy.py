from __future__ import annotations

import random
from dataclasses import dataclass, field

from .types import ActionType, ActionUtilities
from .utility import entropy, softmax


@dataclass
class MixedStrategyPolicy:
    seed: int = 7
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def action_distribution(self, action_utilities: ActionUtilities, temperature: float) -> dict[ActionType, float]:
        actions = list(action_utilities.keys())
        scores = [action_utilities[a] for a in actions]
        probs = softmax(scores, temperature=temperature)
        return {a: p for a, p in zip(actions, probs)}

    def choose(self, action_utilities: ActionUtilities, temperature: float) -> tuple[ActionType, dict[ActionType, float]]:
        dist = self.action_distribution(action_utilities, temperature)
        threshold = self.rng.random()
        acc = 0.0
        for action, prob in dist.items():
            acc += prob
            if threshold <= acc:
                return action, dist
        return next(reversed(dist)), dist

    @staticmethod
    def exploitability_index(dist: dict[ActionType, float]) -> float:
        # Concentracion de la politica (0 = totalmente uniforme).
        uniform = 1.0 / max(len(dist), 1)
        return sum(abs(p - uniform) for p in dist.values()) / 2.0

    @staticmethod
    def policy_entropy(dist: dict[ActionType, float]) -> float:
        return entropy(dist.values())
