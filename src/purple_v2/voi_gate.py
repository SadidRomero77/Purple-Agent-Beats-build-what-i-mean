from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig
from .types import Observation
from .utility import clamp


@dataclass
class VOIGate:
    config: AgentConfig

    def value_of_information(self, obs: Observation, uncertainty: float) -> float:
        ambiguity = clamp(obs.ambiguity)
        uncertainty = clamp(uncertainty)

        # Error esperado si no preguntamos.
        expected_error_without = ambiguity * uncertainty

        # Ganancia estimada al desambiguar.
        disambiguation_gain = 0.45 * ambiguity + 0.2 * (1 - obs.format_stability)
        expected_error_with = max(0.0, expected_error_without - disambiguation_gain)

        benefit = expected_error_without - expected_error_with
        voi = benefit - self.config.ask_cost
        return voi

    def should_ask(self, obs: Observation, uncertainty: float) -> bool:
        return self.value_of_information(obs, uncertainty) > self.config.voi_threshold
