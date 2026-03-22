from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purple_v2.meta_reasoner import MetaReasoner
from purple_v2.types import MentalState


class MetaReasonerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reasoner = MetaReasoner()

    def test_flags_high_uncertainty_and_overconfidence(self) -> None:
        state = MentalState(
            beliefs={"context_uncertainty": 0.9},
            desires={"maximize_score": 1.0},
            intentions=["act"],
            assumptions=["incertidumbre_alta"],
        )
        findings = self.reasoner.audit(state, epistemic_tag="inferencia_fuerte")
        self.assertTrue(any("inconsistencia" in f for f in findings))


if __name__ == "__main__":
    unittest.main()
