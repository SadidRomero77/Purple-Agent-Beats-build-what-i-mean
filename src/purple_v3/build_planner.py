"""BAML-powered build planner: decomposes instructions into atomic build steps.

Key improvements over v2:
1. Uses BAML for structured output (type-safe BuildStep parsing)
2. Worked examples from actual stimulus CSV data
3. Accepts history context for cross-round learning
4. Fallback to direct OpenAI call if BAML fails
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .grid import Grid, GridConfig
from .prompt_enricher import enrich_prompt

logger = logging.getLogger(__name__)


@dataclass
class BuildStep:
    """A single atomic build action."""
    action: str  # stack, place, place_relative, extend_row, place_at_corners
    color: str
    count: int | str = 1
    position: Dict[str, Any] = field(default_factory=dict)
    direction: str = ""
    reference: str = ""


# Map BAML enum values to internal action strings
_ACTION_MAP = {
    "Stack": "stack",
    "Place": "place",
    "PlaceRelative": "place_relative",
    "ExtendRow": "extend_row",
    "PlaceAtCorners": "place_at_corners",
}

_DIRECTION_MAP = {
    "Right": "right",
    "Left": "left",
    "Front": "front",
    "Behind": "behind",
}


class BuildPlanner:
    """Decomposes instructions into build steps using BAML + LLM."""

    def __init__(self, client: Any = None, model: str = "", config: GridConfig | None = None):
        self._client = client
        self._model = model
        self._config = config or GridConfig()

    async def decompose(
        self,
        instruction: str,
        start_grid: Grid,
        speaker: str = "",
        structure_hint: str = "",
        correction_hint: str = "",
        history: list[dict] | None = None,
    ) -> List[BuildStep] | None:
        """Decompose instruction into build steps using BAML."""
        # Build history context
        history_context = self._format_history(history)

        # Build enrichment
        enrichment = enrich_prompt(instruction)

        # Try BAML first
        try:
            steps = await self._decompose_baml(
                instruction, start_grid, structure_hint,
                history_context, enrichment, correction_hint,
            )
            if steps:
                return steps
        except Exception as exc:
            logger.warning("BAML planner failed: %s", exc)

        # Fallback to direct OpenAI JSON mode
        if self._client:
            return await self._decompose_openai(
                instruction, start_grid, speaker,
                structure_hint, correction_hint, history_context, enrichment,
            )

        return None

    async def _decompose_baml(
        self,
        instruction: str,
        start_grid: Grid,
        structure_hint: str,
        history_context: str,
        enrichment: str,
        correction_hint: str,
    ) -> List[BuildStep] | None:
        """Use BAML for structured decomposition."""
        from .baml_client.baml_client import b

        result = await b.DecomposeBuildInstruction(
            instruction=instruction,
            start_structure=start_grid.to_str() or "(empty)",
            structure_analysis=structure_hint or "No existing structure.",
            history_context=history_context,
            enrichment=enrichment or "",
            correction_hint=correction_hint or "",
        )

        if not result.steps:
            return None

        steps = []
        for s in result.steps:
            action = _ACTION_MAP.get(s.action.value if hasattr(s.action, 'value') else str(s.action), str(s.action).lower())
            direction = ""
            if s.direction is not None:
                direction = _DIRECTION_MAP.get(
                    s.direction.value if hasattr(s.direction, 'value') else str(s.direction),
                    str(s.direction).lower()
                )

            count: int | str = s.count
            if count == 0:
                count = "Uncounted"

            color = s.color
            if color.lower() in ("uncolored", "unknown", "unspecified"):
                color = "Uncolored"

            steps.append(BuildStep(
                action=action,
                color=color,
                count=count,
                position={"x": s.position.x, "z": s.position.z},
                direction=direction,
                reference=s.reference or "",
            ))

        logger.info("BAML planner: %d steps", len(steps))
        return steps

    async def _decompose_openai(
        self,
        instruction: str,
        start_grid: Grid,
        speaker: str,
        structure_hint: str,
        correction_hint: str,
        history_context: str,
        enrichment: str,
    ) -> List[BuildStep] | None:
        """Fallback: direct OpenAI call with JSON mode."""
        parts = []
        if structure_hint:
            parts.append(f"EXISTING STRUCTURE ANALYSIS:\n{structure_hint}")
        parts.append(f"START_STRUCTURE: {start_grid.to_str() or '(empty)'}")
        if speaker:
            parts.append(f"SPEAKER: {speaker}")
        if history_context:
            parts.append(history_context)
        parts.append(f"INSTRUCTION: {instruction}")
        if correction_hint:
            parts.append(f"\nCORRECTION NEEDED:\n{correction_hint}")

        user_prompt = "\n".join(parts)

        system = _FALLBACK_SYSTEM_PROMPT
        if enrichment:
            system += enrichment

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            return self._parse_steps(content)
        except Exception as exc:
            logger.warning("OpenAI planner fallback failed: %s", exc)
            return None

    def _format_history(self, history: list[dict] | None) -> str:
        """Format conversation history for the LLM."""
        if not history:
            return ""

        parts = ["PREVIOUS ROUNDS (learn from these):"]
        for entry in history[-6:]:  # Last 6 entries
            if entry["type"] == "instruction":
                parts.append(f"  Instruction: {entry['content'][:200]}")
            elif entry["type"] == "response":
                parts.append(f"  My response: {entry['content'][:200]}")
            elif entry["type"] == "feedback":
                parts.append(f"  Feedback: {entry['content'][:200]}")
        return "\n".join(parts)

    def _parse_steps(self, content: str) -> List[BuildStep] | None:
        """Parse LLM JSON response into BuildStep objects."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    return None
            else:
                return None

        steps_data = data.get("steps", [])
        if not steps_data:
            return None

        steps = []
        for s in steps_data:
            action = s.get("action", "place")
            color = s.get("color", "Uncolored")
            count = s.get("count", 1)
            position = s.get("position", {})
            direction = s.get("direction", "")
            reference = s.get("reference", "")

            if isinstance(count, str) and count.lower() not in ("uncounted", "unknown", "unspecified", "?"):
                try:
                    count = int(count)
                except ValueError:
                    count = "Uncounted"

            steps.append(BuildStep(
                action=action,
                color=color,
                count=count,
                position=position,
                direction=direction,
                reference=reference,
            ))

        return steps


_FALLBACK_SYSTEM_PROMPT = r"""You are a precise block-building planner on a 9x9 grid.

COORDINATE SYSTEM:
- Grid is x-z plane. Origin (0,0) is center ("middle square").
- X: left (-x) to right (+x). Valid: [-400,-300,-200,-100,0,100,200,300,400]
- Z: back (-z) to front (+z). Valid: [-400,-300,-200,-100,0,100,200,300,400]
- Y: vertical. Ground=50. Each block +100. Valid: [50,150,250,350,450]

DIRECTIONS:
- "right" = +x | "left" = -x | "in front of" = +z | "behind" = -z | "on top" = +y

CORNERS: bottom-left=(-400,400), bottom-right=(400,400), top-left=(-400,-400), top-right=(400,-400)

STEP TYPES:
1. "stack" - N blocks vertically at (x,z). {"action":"stack","color":"Red","count":3,"position":{"x":0,"z":0}}
2. "place" - Single block. {"action":"place","color":"Blue","count":1,"position":{"x":100,"z":200}}
3. "place_relative" - Relative to reference. {"action":"place_relative","color":"Green","count":1,"reference":"Red_at_0_0","direction":"right"}
4. "extend_row" - N blocks in line. {"action":"extend_row","color":"Yellow","count":3,"position":{"x":0,"z":0},"direction":"right"}
5. "place_at_corners" - All corners. {"action":"place_at_corners","color":"Red","count":4}

RULES:
- Only x,z in position. Y auto-managed.
- "Uncolored" if no color specified. "Uncounted" if no count.
- Chain refs: use NEW position from previous step.
- Row "from middle going right" starts AT x=0.

OUTPUT: {"steps": [...]} — JSON only, no explanation.
"""
