from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purple_v2.belief_state import BeliefState
from purple_v2.config import AgentConfig
from purple_v2.types import Observation, OpponentType


class BeliefStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = BeliefState(AgentConfig())

    def test_beliefs_sum_to_one(self) -> None:
        obs = Observation(ambiguity=0.6, compliance=0.6, aggression=0.3, deception_signal=0.2, format_stability=0.8)
        beliefs = self.state.update(obs)
        self.assertAlmostEqual(sum(beliefs.values()), 1.0, places=7)

    def test_adversarial_signal_shifts_posterior(self) -> None:
        obs = Observation(ambiguity=0.4, compliance=0.2, aggression=0.9, deception_signal=0.8, format_stability=0.3)
        beliefs = self.state.update(obs)
        top = max(beliefs, key=beliefs.get)
        self.assertEqual(top, OpponentType.ADVERSARIAL)


if __name__ == "__main__":
    unittest.main()
