#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purple_v2.agent import PurpleAgentV2
from purple_v2.config import AgentConfig
from purple_v2.types import AgentContext, Observation


def main() -> None:
    agent = PurpleAgentV2(AgentConfig())

    rounds = [
        Observation(ambiguity=0.7, compliance=0.4, aggression=0.2, deception_signal=0.3, format_stability=0.8),
        Observation(ambiguity=0.5, compliance=0.35, aggression=0.75, deception_signal=0.7, format_stability=0.55),
        Observation(ambiguity=0.9, compliance=0.2, aggression=0.5, deception_signal=0.6, format_stability=0.25),
    ]

    for idx, obs in enumerate(rounds, start=1):
        ctx = AgentContext(round_index=idx, uncertainty=obs.ambiguity)
        decision, debug = agent.decide(obs, ctx)
        payload = agent.build_payload(
            prompt="Construye una respuesta competitiva robusta",
            decision=decision,
            epistemic_status=debug["epistemic_status"],
        )

        print(f"\n=== Ronda {idx} ===")
        print("Decision:", decision.action.value)
        print("Payload:", json.dumps(payload, ensure_ascii=False, indent=2))
        print("Debug:", json.dumps(debug, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
