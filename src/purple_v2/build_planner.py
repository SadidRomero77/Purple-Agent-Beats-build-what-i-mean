"""LLM-based build planner: decomposes natural language instructions into atomic build steps."""
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


SYSTEM_PROMPT = r"""You are a precise block-building planner on a 9x9 grid.

COORDINATE SYSTEM:
- The grid is the x-z plane. Origin (0,0) is the center ("middle square" / "highlighted square").
- X-axis: left (-x) to right (+x). Valid: [-400,-300,-200,-100,0,100,200,300,400]
- Z-axis: back (-z) to front (+z). Valid: [-400,-300,-200,-100,0,100,200,300,400]
- Y-axis: vertical height. Ground=50. Each block adds +100.
  Valid y: [50, 150, 250, 350, 450] (max 5 blocks high)

DIRECTIONS (CRITICAL - get these right):
- "to the right" / "right" = +x (increasing x)
- "to the left" / "left" = -x (decreasing x)
- "in front of" / "front" = +z (increasing z)
- "behind" / "back" = -z (decreasing z)
- "on top of" = +y (increasing y, same x,z)

CORNERS (at ground level y=50):
- bottom-left = (-400, 50, 400)
- bottom-right = (400, 50, 400)
- top-left = (-400, 50, -400)
- top-right = (400, 50, -400)

YOUR TASK:
Decompose the instruction into atomic JSON build steps. Output ONLY valid JSON.

STEP TYPES:
1. "stack" - N blocks vertically at (x,z). y is auto-computed.
   {"action":"stack", "color":"Red", "count":3, "position":{"x":0,"z":0}}

2. "place" - Single block at absolute position.
   {"action":"place", "color":"Blue", "count":1, "position":{"x":100,"z":200}}

3. "place_relative" - Block relative to a reference.
   {"action":"place_relative", "color":"Green", "count":1,
    "reference":"Red_block_at_0_0", "direction":"right"}

4. "extend_row" - N blocks in a line from a start position.
   {"action":"extend_row", "color":"Yellow", "count":3,
    "position":{"x":0,"z":0}, "direction":"right"}

5. "place_at_corners" - Blocks at grid corners.
   {"action":"place_at_corners", "color":"Red", "count":4}

RULES:
- Only specify x,z in position. Y is auto-managed (ground or stacking).
- Use "Uncolored" if the instruction does NOT specify a color for some blocks.
- Use "Uncounted" (as string) for count if the instruction does NOT specify how many.
- For extend_row, "position" is WHERE THE ROW STARTS and "direction" says which way it goes.
- For chain references ("the green one", "the block you just placed"): reference the
  NEWLY PLACED position from a previous step, not the original position.

WORKED EXAMPLES:

Example 1: "Place a red block in each corner of the grid."
START_STRUCTURE: (empty)
STEPS:
{"steps":[
  {"action":"place","color":"Red","count":1,"position":{"x":-400,"z":400}},
  {"action":"place","color":"Red","count":1,"position":{"x":400,"z":400}},
  {"action":"place","color":"Red","count":1,"position":{"x":-400,"z":-400}},
  {"action":"place","color":"Red","count":1,"position":{"x":400,"z":-400}}
]}

Example 2: "Stack three blue blocks on the highlighted square."
START_STRUCTURE: (empty)
STEPS:
{"steps":[
  {"action":"stack","color":"Blue","count":3,"position":{"x":0,"z":0}}
]}

Example 3: "Build a row of four green blocks starting from the middle square going right."
START_STRUCTURE: (empty)
STEPS:
{"steps":[
  {"action":"extend_row","color":"Green","count":4,"position":{"x":0,"z":0},"direction":"right"}
]}
Note: starts AT x=0, not x=100.

Example 4: "Stack two red blocks directly in front of the green blocks."
START_STRUCTURE: Green,0,50,0;Green,0,150,0
STEPS:
{"steps":[
  {"action":"stack","color":"Red","count":2,"position":{"x":0,"z":100}}
]}
Note: "in front of" = +z. Green is at z=0, so red goes at z=100.

Example 5: "Place a yellow block to the left of the red block, then place a blue block to the left of the yellow one."
START_STRUCTURE: Red,0,50,0
STEPS:
{"steps":[
  {"action":"place","color":"Yellow","count":1,"position":{"x":-100,"z":0}},
  {"action":"place","color":"Blue","count":1,"position":{"x":-200,"z":0}}
]}
Note: Chain reference! Blue goes left of WHERE yellow was placed (-100), not left of red.

Example 6: "Extend the row by two blocks to the right, then stack a block on each end."
START_STRUCTURE: Red,-100,50,0;Red,0,50,0;Red,100,50,0
STEPS:
{"steps":[
  {"action":"extend_row","color":"Red","count":2,"position":{"x":200,"z":0},"direction":"right"},
  {"action":"stack","color":"Red","count":1,"position":{"x":-100,"z":0}},
  {"action":"stack","color":"Red","count":1,"position":{"x":300,"z":0}}
]}
Note: After extending right by 2, row is [-100,0,100,200,300]. Ends are -100 and 300 (NOT 100!).

Example 7: "Place nine blocks along the left edge of the grid."
START_STRUCTURE: (empty)
STEPS:
{"steps":[
  {"action":"extend_row","color":"Uncolored","count":9,"position":{"x":-400,"z":-400},"direction":"front"}
]}
Note: Left edge = x=-400, varying z from -400 to 400.

Example 8: "Stack blocks behind the rightmost green block."
START_STRUCTURE: Green,-200,50,0;Green,-200,150,0;Green,100,50,0;Green,100,150,0
STEPS:
{"steps":[
  {"action":"stack","color":"Uncolored","count":"Uncounted","position":{"x":100,"z":-100}}
]}
Note: Rightmost green at x=100. Behind = -z. So position is (100, -100). Count/color unspecified.

Example 9: "Extend the longer side of the L-shape by two blocks."
START_STRUCTURE: Blue,-100,50,0;Blue,0,50,0;Blue,100,50,0;Blue,100,50,-100
L-shape with corner at (100,0). Horizontal arm (3 blocks, longer) and vertical arm (2 blocks).
STEPS:
{"steps":[
  {"action":"extend_row","color":"Blue","count":2,"position":{"x":-200,"z":0},"direction":"left"}
]}
Note: Extend the LONGER arm (x-axis, 3 blocks) AWAY from corner. Corner at x=100, arm goes left, so extend from x=-200 going left.

Example 10: "Extend the stem of the T-shape by two blocks."
START_STRUCTURE: Red,-100,50,0;Red,0,50,0;Red,100,50,0;Red,0,50,-100
T-shape: crossbar along x at z=0, stem going back (z=-100) from junction (0,0).
STEPS:
{"steps":[
  {"action":"extend_row","color":"Red","count":2,"position":{"x":0,"z":-200},"direction":"behind"}
]}
Note: Stem tip at (0,-100). Extend AWAY from junction, so continue behind from (0,-200).

Example 11: "Build a horizontal row of three purple blocks in front of the yellow stack, then put a green block on top of the first purple block you placed."
START_STRUCTURE: Yellow,200,50,100;Yellow,200,150,100;Yellow,200,250,100
STEPS:
{"steps":[
  {"action":"extend_row","color":"Purple","count":3,"position":{"x":200,"z":200},"direction":"right"},
  {"action":"stack","color":"Green","count":1,"position":{"x":200,"z":200}}
]}
Note: "in front of" yellow at z=100 means z=200. "first purple block you placed" = start of the row at (200,200).

Example 12: "Place a blue block to the right of the red block. Then place a green block behind the blue one."
START_STRUCTURE: Red,0,50,0
STEPS:
{"steps":[
  {"action":"place","color":"Blue","count":1,"position":{"x":100,"z":0}},
  {"action":"place","color":"Green","count":1,"position":{"x":100,"z":-100}}
]}
Note: Blue placed at (100,0). Green goes behind THE BLUE ONE at (100,-100), not behind red.

OUTPUT FORMAT:
Respond with ONLY a JSON object: {"steps": [...]}
No explanation, no markdown, no extra text. Just the JSON.
"""


class BuildPlanner:
    """Decomposes instructions into build steps using LLM."""

    def __init__(self, client: Any, model: str, config: GridConfig):
        self._client = client
        self._model = model
        self._config = config

    async def decompose(
        self,
        instruction: str,
        start_grid: Grid,
        speaker: str = "",
        structure_hint: str = "",
        correction_hint: str = "",
    ) -> List[BuildStep] | None:
        """Decompose instruction into build steps."""
        # Build the user prompt
        parts = []
        if structure_hint:
            parts.append(f"EXISTING STRUCTURE ANALYSIS:\n{structure_hint}")
        parts.append(f"START_STRUCTURE: {start_grid.to_str() or '(empty)'}")
        if speaker:
            parts.append(f"SPEAKER: {speaker}")
        parts.append(f"INSTRUCTION: {instruction}")
        if correction_hint:
            parts.append(f"\nCORRECTION NEEDED:\n{correction_hint}")

        user_prompt = "\n".join(parts)

        # Add contextual enrichment
        enrichment = enrich_prompt(instruction)
        system = SYSTEM_PROMPT
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
            logger.warning("Planner LLM call failed: %s", exc)
            return None

    def _parse_steps(self, content: str) -> List[BuildStep] | None:
        """Parse LLM JSON response into BuildStep objects."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse planner output: %s", content[:200])
                    return None
            else:
                logger.warning("No valid JSON in planner output: %s", content[:200])
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

            # Normalize count
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
