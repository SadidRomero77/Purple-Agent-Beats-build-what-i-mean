from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purple_v2.config import AgentConfig
from purple_v2.types import Observation
from purple_v2.voi_gate import VOIGate


class VOIGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = VOIGate(AgentConfig())

    def test_should_ask_when_ambiguity_and_uncertainty_high(self) -> None:
        obs = Observation(ambiguity=0.95, compliance=0.4, aggression=0.2, deception_signal=0.1, format_stability=0.2)
        self.assertTrue(self.gate.should_ask(obs, uncertainty=0.9))

    def test_should_not_ask_when_signal_is_clear(self) -> None:
        obs = Observation(ambiguity=0.1, compliance=0.8, aggression=0.1, deception_signal=0.1, format_stability=0.95)
        self.assertFalse(self.gate.should_ask(obs, uncertainty=0.2))


if __name__ == "__main__":
    unittest.main()
