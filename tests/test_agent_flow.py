from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purple_v2.agent import PurpleAgentV2
from purple_v2.config import AgentConfig
from purple_v2.types import AgentContext, Observation


class AgentFlowTests(unittest.TestCase):
    def test_end_to_end_payload_is_valid(self) -> None:
        agent = PurpleAgentV2(AgentConfig())

        obs = Observation(
            ambiguity=0.8,
            compliance=0.35,
            aggression=0.6,
            deception_signal=0.55,
            format_stability=0.4,
        )
        ctx = AgentContext(round_index=1, uncertainty=0.8)

        decision, debug = agent.decide(obs, ctx)
        payload = agent.build_payload(
            prompt="Resuelve el escenario con estrategia robusta",
            decision=decision,
            epistemic_status=debug["epistemic_status"],
        )

        errors = agent.validator.validate(payload)
        self.assertEqual(errors, [])
        self.assertIn("action", payload)
        self.assertIn("confidence", payload)


if __name__ == "__main__":
    unittest.main()
