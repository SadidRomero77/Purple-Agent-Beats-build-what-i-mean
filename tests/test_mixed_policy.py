from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purple_v2.mixed_policy import MixedStrategyPolicy
from purple_v2.types import ActionType


class MixedPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MixedStrategyPolicy(seed=11)

    def test_distribution_sums_to_one(self) -> None:
        utilities = {
            ActionType.ANSWER_DIRECT: 0.7,
            ActionType.ASK_CLARIFICATION: 0.5,
            ActionType.PROBE_OPPONENT: 0.6,
            ActionType.HEDGE_SAFETY: 0.4,
            ActionType.SAFE_FALLBACK: 0.2,
        }
        dist = self.policy.action_distribution(utilities, temperature=1.0)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=7)

    def test_high_temperature_is_less_concentrated(self) -> None:
        utilities = {
            ActionType.ANSWER_DIRECT: 1.2,
            ActionType.ASK_CLARIFICATION: 0.2,
            ActionType.PROBE_OPPONENT: 0.1,
            ActionType.HEDGE_SAFETY: -0.1,
            ActionType.SAFE_FALLBACK: -0.2,
        }
        cold = self.policy.action_distribution(utilities, temperature=0.5)
        hot = self.policy.action_distribution(utilities, temperature=1.2)

        cold_ex = self.policy.exploitability_index(cold)
        hot_ex = self.policy.exploitability_index(hot)
        self.assertLess(hot_ex, cold_ex)


if __name__ == "__main__":
    unittest.main()
