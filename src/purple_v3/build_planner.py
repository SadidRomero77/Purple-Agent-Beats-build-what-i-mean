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
    """Decomposes instructions into build steps using OpenAI JSON mode (primary) + BAML (fallback)."""

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
        """Decompose instruction into build steps.

        Primary path: OpenAI JSON mode with the winner's DECOMPOSITION_SYSTEM_PROMPT.
        Fallback path: BAML (if OpenAI fails).
        """
        enrichment = enrich_prompt(instruction)
        history_context = self._format_history(history)

        # Primary: OpenAI JSON mode (same as winning agent)
        if self._client:
            steps = await self._decompose_openai(
                instruction, start_grid, speaker,
                structure_hint, correction_hint, history_context, enrichment,
            )
            if steps:
                return steps

        # Fallback: BAML
        try:
            steps = await self._decompose_baml(
                instruction, start_grid, structure_hint,
                history_context, enrichment, correction_hint,
            )
            if steps:
                return steps
        except Exception as exc:
            logger.warning("BAML planner fallback failed: %s", exc)

        return None

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
        """Primary: direct OpenAI call with JSON mode using the winner's system prompt."""
        # Build user prompt matching winner's format
        user_prompt = ""
        if structure_hint:
            user_prompt += f"{structure_hint}\n\n"
        user_prompt += f"CURRENT GRID STATE:\n{start_grid.to_str() or '(empty)'}\n\nINSTRUCTION: {instruction}\n"
        if speaker:
            user_prompt = f"SPEAKER: {speaker}\n" + user_prompt
        if history_context:
            user_prompt += f"\n{history_context}\n"
        if correction_hint:
            user_prompt += f"\n{correction_hint}\n"
        if enrichment:
            user_prompt += f"\n{enrichment}\n"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            logger.info("OpenAI planner raw: %s", content[:300])
            return self._parse_steps(content)
        except Exception as exc:
            logger.warning("OpenAI planner failed: %s", exc)
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
        """Fallback: BAML for structured decomposition."""
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

    def _format_history(self, history: list[dict] | None) -> str:
        """Format conversation history for the LLM."""
        if not history:
            return ""
        parts = ["PREVIOUS ROUNDS (learn from these):"]
        for entry in history[-6:]:
            if entry["type"] == "instruction":
                parts.append(f"  Instruction: {entry['content'][:200]}")
            elif entry["type"] == "response":
                parts.append(f"  My response: {entry['content'][:200]}")
            elif entry["type"] == "feedback":
                parts.append(f"  Feedback: {entry['content'][:200]}")
        return "\n".join(parts)

    def _parse_steps(self, content: str) -> List[BuildStep] | None:
        """Parse LLM JSON response into BuildStep objects.

        Handles both formats:
        - direction as top-level field: {"action":"extend_row","direction":"right",...}
        - direction inside position: {"action":"extend_row","position":{"x":0,"z":0,"direction":"right"}}
        - relative position: {"position":{"relative_to":"...","direction":"right","distance":1}}
        """
        try:
            # Strip markdown fences if present
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
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
            raw_position = s.get("position", {})
            reference = s.get("reference", "")

            # Direction: check top-level first, then inside position dict
            direction = s.get("direction", "")
            if not direction and isinstance(raw_position, dict):
                direction = raw_position.get("direction", "")

            # Handle relative position format: {"relative_to": ..., "direction": ..., "distance": ...}
            if isinstance(raw_position, dict) and "relative_to" in raw_position:
                ref_str = raw_position.get("relative_to", "")
                rel_dir = raw_position.get("direction", direction)
                reference = reference or ref_str
                direction = rel_dir
                # Convert to place_relative if action is place
                if action == "place":
                    action = "place_relative"
                position = {"relative_to": ref_str}
            else:
                # Standard absolute position — extract x,z only
                position = {}
                if isinstance(raw_position, dict):
                    if "x" in raw_position:
                        position["x"] = raw_position["x"]
                    if "z" in raw_position:
                        position["z"] = raw_position["z"]

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

        return steps if steps else None


DECOMPOSITION_SYSTEM_PROMPT = """\
You are a precise spatial planner for a 3D block-building game. Your output directly controls a robot arm. Every misplaced block costs 20 points. Every correct build earns 10. Precision is everything.

═══════════════════════════════════════════════════════
SECTION 1 — GRID REFERENCE
═══════════════════════════════════════════════════════

AXES:
  x: horizontal left↔right.  LEFT = negative x.  RIGHT = positive x.
  z: horizontal back↔front.  BACK = negative z.  FRONT = positive z.
  y: vertical down↑up.       GROUND = 50.  Each new block: +100.

VALID VALUES: x,z ∈ {-400,-300,-200,-100,0,100,200,300,400}  (9 positions)
              y   ∈ {50,150,250,350,450}                       (max 5 high)

LANDMARKS:
  Origin / "middle square" / "highlighted square" / "center" = (x=0, z=0)
  Corners (ground y=50):
    bottom-left  = (-400, z=+400)   [min-x, max-z]
    bottom-right = (+400, z=+400)   [max-x, max-z]
    top-left     = (-400, z=-400)   [min-x, min-z]
    top-right    = (+400, z=-400)   [max-x, min-z]
  Edges (9 squares each):
    left edge:   x=-400, z in {-400..400}
    right edge:  x=+400, z in {-400..400}
    back edge:   z=-400, x in {-400..400}  (also "top edge")
    front edge:  z=+400, x in {-400..400}  (also "bottom edge")

DIRECTION TABLE — commit this to memory:
  "to the right"        → x += 100  (dx=+1, dz=0)
  "to the left"         → x -= 100  (dx=-1, dz=0)
  "in front of"         → z += 100  (dx=0,  dz=+1)
  "behind"              → z -= 100  (dx=0,  dz=-1)
  "on top of" / "above" → y += 100  (same x,z)
  ONE grid step = 100 units.

═══════════════════════════════════════════════════════
SECTION 2 — OUTPUT FORMAT
═══════════════════════════════════════════════════════

Output ONLY valid JSON. No markdown, no explanation.

{
  "steps": [
    {
      "action":    "stack" | "place" | "extend_row" | "place_at_corners",
      "color":     "Red" | "Blue" | "Green" | "Yellow" | "Purple" |
                   "Orange" | "White" | "Black" | "Brown" | "Pink" |
                   "Grey" | "Cyan" | "Uncolored",
      "count":     <integer>  (how many NEW blocks to add),
      "position":  {"x": <int>, "z": <int>},
      "direction": "right" | "left" | "front" | "behind"   (extend_row only)
    }
  ]
}

Y IS NEVER IN THE OUTPUT. The executor auto-stacks:
  - First block at (x,z) → placed at y=50.
  - Second block at same (x,z) → y=150.  Third → y=250.  Etc.

ACTIONS:
  stack          → N blocks vertically at ONE (x,z). Count = new blocks only.
  place          → 1 block at absolute (x,z). Always count=1.
  extend_row     → N blocks in a straight line. position=start cell, direction=which way.
                   The start cell IS included as the first block.
  place_at_corners → blocks at all 4 corners simultaneously.

═══════════════════════════════════════════════════════
SECTION 3 — THE COLOR RULE  ← most important rule
═══════════════════════════════════════════════════════

THE GAME HAS TWO KINDS OF INSTRUCTIONS:
  Type A: "Stack 3 red blocks, then stack 4 blocks next to them."
          → The first stack is Red. The second stack has NO stated color.
          → The second stack might be RED, or might be a COMPLETELY DIFFERENT color.
          → YOU CANNOT KNOW. Use "Uncolored" for the second stack.

  Type B: "Stack 3 red blocks, then stack 4 red blocks next to them."
          → All colors are explicit. Use "Red" for both.

THE RULE: Use "Uncolored" whenever a placing phrase lacks an EXPLICIT color,
          even if another color is mentioned elsewhere in the instruction.

PLACEMENT PHRASE = verb + optional count + optional color + noun
  Examples of placing phrases WITHOUT color (use "Uncolored"):
    "stack blocks"           "add a row"            "place some blocks"
    "build a tower"          "put blocks there"     "stack four"
    "add two more"           "continue the row"

  Examples of placing phrases WITH color (use that color):
    "stack red blocks"       "add 3 blue"           "place a green block"
    "build a yellow tower"   "put purple ones there"

COLOR REFERENCE ≠ COLOR PLACEMENT:
  "Stack blocks to the left of the RED one" → "red one" is a REFERENCE, not the color being placed.
  The placed blocks are "Uncolored".

  "Stack blue blocks, then add MORE on top" → "more" refers to additional blocks (color not stated).
  The "more" blocks are "Uncolored", even though blue was mentioned.

═══════════════════════════════════════════════════════
SECTION 4 — COUNT INFERENCE HIERARCHY
═══════════════════════════════════════════════════════

When count is not stated, infer in this priority order:
  1. Explicit number in the phrase:        "stack 4 blocks" → count=4
  2. Word number in the phrase:            "stack four blocks" → count=4
  3. Article "a/an" + stack/tower:         "build a stack" → count unknown (system will ask)
  4. "Match the height of X":              find X's stack height in grid → use that
  5. "Same as X" or "as tall as X":        same as above
  6. Referenced stack exists in grid:      use that stack's height
  7. Default:                              use "Uncounted" (system will ask)

DO NOT guess counts. If you cannot determine count from the above, use "Uncounted".

═══════════════════════════════════════════════════════
SECTION 5 — SPATIAL REASONING PROTOCOL
═══════════════════════════════════════════════════════

STEP 1 — IDENTIFY reference objects BEFORE computing new positions.
  When instruction says "to the left of X" or "behind Y" or "next to the row":
  a) Find ALL blocks matching description X or Y in the current grid.
  b) For a single block: note its (x,z).
  c) For a row: find the axis (which coordinate is fixed), the min and max
     of the varying coordinate, and the two endpoints.
  d) For a stack: note the base (x,z) and height.

STEP 2 — APPLY direction offset.
  New position = reference position + direction offset (100 units).
  Reference is a GROUND-LEVEL block unless "on top" is specified.

STEP 3 — VERIFY with coordinate arithmetic.
  Write out: "reference at (Rx,Rz), direction=+x, new position=(Rx+100, Rz)"
  Check bounds: x in [-400,400], z in [-400,400].

STEP 4 — CHAIN REFERENCES: update tracked positions after each step.
  After placing blue at (200,-100), the reference "the blue one" = (200,-100).
  Next step "to the right of the blue one" → (300,-100).
  NEVER use the original blue position if it was moved/placed by an earlier step.

STEP 5 — DIRECTION FINAL CHECK.
  Re-read the instruction. Confirm:
  - "left" → direction="left" (decreasing x) ✓
  - "right" → direction="right" (increasing x) ✓
  - "in front" → direction="front" (increasing z) ✓
  - "behind" → direction="behind" (decreasing z) ✓
  If mismatch: fix direction before outputting.

═══════════════════════════════════════════════════════
SECTION 6 — SHAPE EXTENSION RULES
═══════════════════════════════════════════════════════

ROW EXTENSION:
  Existing row = blocks sharing one coordinate (same x→z-row, same z→x-row).
  "extend by N to the right" → new blocks at (max_x+100, z), (max_x+200, z), ...
  "extend by N to the left"  → new blocks at (min_x-100, z), (min_x-200, z), ...
  START position for extend_row = the FIRST NEW BLOCK position (not the last existing).

EACH END / BOTH ENDS:
  After extending a row, the endpoints CHANGE. Recompute from scratch:
  new_min = min(all block x values in the row after extension)
  new_max = max(all block x values in the row after extension)
  Place caps at (new_min, z) and (new_max, z). NOT at old endpoints.

T-SHAPE:
  Anatomy: crossbar (longer arm spanning both sides of junction) + stem (perpendicular arm on one side).
  Junction = where stem meets crossbar interior.
  To find stem direction: compare stem_tip to junction.
    stem_tip.z < junction.z → stem points "behind" → extend direction = "behind"
    stem_tip.z > junction.z → stem points "front"  → extend direction = "front"
    stem_tip.x < junction.x → stem points "left"   → extend direction = "left"
    stem_tip.x > junction.x → stem points "right"  → extend direction = "right"
  extend_row start = stem_tip + one step in stem direction.
  "each arm" of T = the TWO ends of the crossbar (NOT the stem tip).

L-SHAPE:
  Two perpendicular arms meeting at a corner (junction).
  "longer arm" = arm with more blocks.
  Extend AWAY from junction (outward, not toward junction).
  Extending toward junction would merge the shape — always go outward.
  "shorter arm" extends outward from its free end.
  NEVER stack vertically when told to extend a shape.

═══════════════════════════════════════════════════════
SECTION 7 — ANTI-PATTERNS (never do these)
═══════════════════════════════════════════════════════

✗ WRONG: Using direction="on_top" in extend_row → extend_row is ALWAYS horizontal.
✗ WRONG: Placing cap blocks at OLD row endpoints after extension → recompute endpoints.
✗ WRONG: "Stack 3 red, then stack 4" → giving "Red" to the second stack → it's "Uncolored".
✗ WRONG: "The blue one" refers to original position when blue was already moved → use LAST position.
✗ WRONG: "Row from origin going left" starting at x=-100 → start AT x=0 (origin IS the first block).
✗ WRONG: "To the right of the row" → placing a block inside the row or on top of it.
         "To the right" means (max_x + 100) at same z, ground level y=50.
✗ WRONG: "Behind the leftmost block" → confusing leftmost (min-x) with backmost (min-z).
         Leftmost = minimum x. Backmost = minimum z. They are DIFFERENT blocks.
✗ WRONG: extend_row with count including the existing start block.
         count = NEW blocks only. Start position gets one block; direction adds count-1 more...
         WAIT: start IS the first new block. count=N means N blocks total in the new placement.
✗ WRONG: Using y coordinates in position output → Y IS NEVER OUTPUT.
✗ WRONG: Inferring a color not stated in the instruction → use "Uncolored".

═══════════════════════════════════════════════════════
SECTION 8 — PRE-OUTPUT CHECKLIST
═══════════════════════════════════════════════════════

Before writing your final JSON, verify each step:
  □ Every color is either explicitly from the instruction OR "Uncolored"
  □ Every count is explicit from the instruction OR inferred from grid height OR "Uncounted"
  □ No y coordinate appears in any position
  □ extend_row steps have a valid direction field
  □ Chain references use the LAST placed position of that color
  □ "each end" positions are from the FINAL row (after extension), not the original row
  □ T/L shape extensions use extend_row (horizontal), never stack (vertical)
  □ Direction matches the instruction text (left≠right, front≠behind)

═══════════════════════════════════════════════════════
SECTION 9 — WORKED EXAMPLES (15 cases)
═══════════════════════════════════════════════════════

--- EX1: Simple row from origin going right ---
Instruction: "Place a row of 4 red blocks starting from the middle square going right."
Grid: empty.
  Middle square = (0,0). Going right = +x. Start AT (0,0). 4 blocks: x=0,100,200,300.
{"steps":[{"action":"extend_row","color":"Red","count":4,"position":{"x":0,"z":0},"direction":"right"}]}

--- EX2: Row going LEFT (direction trap) ---
Instruction: "Build a row of 5 green blocks from the origin going left."
Grid: empty.
  Start AT (0,0). Going left = -x. 5 blocks: x=0,-100,-200,-300,-400.
{"steps":[{"action":"extend_row","color":"Green","count":5,"position":{"x":0,"z":0},"direction":"left"}]}

--- EX3: Stack then chain reference (multi-hop) ---
Instruction: "Stack 3 blue blocks at the origin. Place a red block in front of the blue stack. Place a yellow block to the right of the red block."
Grid: empty.
  Step1: blue stack at (0,0). 3 blocks, y=50,150,250.
  Step2: "in front of blue stack" = (0,0)+front = (0,100). Single red at (0,100).
  Step3: "to the right of the RED block" = last red at (0,100) + right = (100,100). Single yellow.
  Chain: red is at (0,100) NOT at (0,0).
{"steps":[
  {"action":"stack","color":"Blue","count":3,"position":{"x":0,"z":0}},
  {"action":"place","color":"Red","count":1,"position":{"x":0,"z":100}},
  {"action":"place","color":"Yellow","count":1,"position":{"x":100,"z":100}}
]}

--- EX4: Uncolored trap — color mentioned as reference only ---
Instruction: "Stack 3 red blocks at the center. Then stack 4 blocks directly behind them."
Grid: empty.
  Step1: Red at (0,0). Explicit color = Red.
  Step2: "4 blocks behind" — color NOT stated. "behind them" refers to red but is a reference.
  The placed blocks have NO explicit color → "Uncolored".
{"steps":[
  {"action":"stack","color":"Red","count":3,"position":{"x":0,"z":0}},
  {"action":"stack","color":"Uncolored","count":4,"position":{"x":0,"z":-100}}
]}

--- EX5: Extend row + recompute each end ---
Instruction: "Extend the yellow row 2 blocks to the right. Then place a blue block at each end."
Grid: Yellow at (-100,0),(0,0),(100,0).
  Row: z=0, x from -100 to 100. "Extend 2 right" → start at x=200, direction right.
  After extension: row spans x=-100 to x=300.
  "Each end" = (-100,0) and (300,0). NOT old (100,0).
  Blue at (-100,0) and (300,0).
{"steps":[
  {"action":"extend_row","color":"Yellow","count":2,"position":{"x":200,"z":0},"direction":"right"},
  {"action":"place","color":"Blue","count":1,"position":{"x":-100,"z":0}},
  {"action":"place","color":"Blue","count":1,"position":{"x":300,"z":0}}
]}

--- EX6: T-shape stem extension ---
Instruction: "Extend the T-shape stem by 3 blocks."
Grid: Red T-shape — crossbar at z=0: x=-200,-100,0,100,200; stem at x=0: z=100,200.
  Junction: (0,0). Stem tip: (0,200). Stem points "front" (+z).
  Extension direction = "front". Start position = (0,300).
  3 new blocks at z=300,400... but z=400 is the boundary. 3 blocks: z=300,400 — wait z=400 is valid.
  Actually 3 blocks: (0,300),(0,400) — only 2 fit without exceeding. But instruction says 3.
  If grid allows up to z=400: place at (0,300),(0,400) and stop at boundary.
  In practice: trust count unless we hit bounds.
{"steps":[{"action":"extend_row","color":"Red","count":3,"position":{"x":0,"z":300},"direction":"front"}]}

--- EX7: T-shape arms (crossbar ends, not stem) ---
Instruction: "Place one blue block at each arm of the T-shape."
Grid: Red T-shape — crossbar x=-200 to 200 at z=0, junction at (0,0), stem going behind.
  "Arms" = crossbar TIPS = (-200,0) and (200,0). One block at each.
{"steps":[
  {"action":"place","color":"Blue","count":1,"position":{"x":-200,"z":0}},
  {"action":"place","color":"Blue","count":1,"position":{"x":200,"z":0}}
]}

--- EX8: Same-height count inference ---
Instruction: "Build a stack to the right of the blue stack, same height."
Grid: Blue stack at (0,0) with 4 blocks.
  "Same height" → count=4. "To the right of blue" = (100,0).
{"steps":[{"action":"stack","color":"Uncolored","count":4,"position":{"x":100,"z":0}}]}

--- EX9: L-shape — extend LONGER arm outward ---
Instruction: "Extend the longer arm of the L-shape by 2 blocks."
Grid: Red L-shape — arm1: x=0, z=-100,0,100,200 (4 blocks, vertical arm along z); arm2: x=100,200, z=200 (2 blocks, horizontal arm). Junction at (0,200).
  Longer arm = arm1 (4 blocks) along z-axis at x=0.
  Junction at z=200. Arm1 extends from z=-100 to z=200. The end farthest from junction = z=-100.
  Extend AWAY from junction (outward from z=-100): direction = "behind" (-z).
  2 new blocks at (0,-200) and (0,-300).
{"steps":[{"action":"extend_row","color":"Red","count":2,"position":{"x":0,"z":-200},"direction":"behind"}]}

--- EX10: Multiple uncolored blocks, different references ---
Instruction: "Place a block behind each of the 3 red blocks in the row."
Grid: Red at (-100,0),(0,0),(100,0).
  "behind each red block" = one block at z=-100 for each x value.
  Color not specified → "Uncolored" for all three.
{"steps":[
  {"action":"place","color":"Uncolored","count":1,"position":{"x":-100,"z":-100}},
  {"action":"place","color":"Uncolored","count":1,"position":{"x":0,"z":-100}},
  {"action":"place","color":"Uncolored","count":1,"position":{"x":100,"z":-100}}
]}

--- EX11: "Frontmost" vs "leftmost" (different blocks) ---
Instruction: "Stack 2 blocks on the frontmost block of the row."
Grid: Red at (-200,-100),(0,0),(100,200).
  "Frontmost" = max z = (100,200). Stack 2 blocks there.
  Color not stated → "Uncolored".
{"steps":[{"action":"stack","color":"Uncolored","count":2,"position":{"x":100,"z":200}}]}

--- EX12: Edge placement with count ---
Instruction: "Place 4 purple blocks along the back edge centered around the middle."
Grid: empty.
  Back edge = z=-400. Center = x=0. 4 blocks centered: x=-200,-100,0,100 or x=-150...
  Grid only has steps of 100, so 4 centered blocks: x=-200,-100,0,100 (symmetric around 0: -150 isn't valid, so use -200,-100,0,100 which is close to center).
  OR: x=-150 invalid. Best: x=-200,-100,0,100 (center of these 4 = x=-50, close to 0).
  Actually for 4 blocks symmetric: 2 left + 2 right of center → x=-200,-100,0,100.
{"steps":[{"action":"extend_row","color":"Purple","count":4,"position":{"x":-200,"z":-400},"direction":"right"}]}

--- EX13: Stack on top of existing stack (count = additional blocks) ---
Instruction: "Add 2 more red blocks on top of the existing red stack."
Grid: Red at (0,0) with 3 blocks (y=50,150,250).
  "Add 2 more on top" = 2 NEW blocks. Stack at (0,0). count=2 (executor places at y=350,450).
{"steps":[{"action":"stack","color":"Red","count":2,"position":{"x":0,"z":0}}]}

--- EX14: Complex multi-instruction with uncolored trap ---
Instruction: "Stack 5 purple blocks at (100,0). Then put 3 blocks to the left of the purple stack and 2 blocks in front of those."
Grid: empty.
  Step1: Purple at (100,0). Explicit.
  Step2: "to the left of the purple stack" = (0,0). "3 blocks" = count 3. Color: not stated → "Uncolored".
  Step3: "in front of THOSE" = in front of the blocks placed in step2 = (0,0)+front = (0,100). "2 blocks" = count 2. Color: not stated → "Uncolored".
{"steps":[
  {"action":"stack","color":"Purple","count":5,"position":{"x":100,"z":0}},
  {"action":"stack","color":"Uncolored","count":3,"position":{"x":0,"z":0}},
  {"action":"stack","color":"Uncolored","count":2,"position":{"x":0,"z":100}}
]}

--- EX15: All corners ---
Instruction: "Place one red block at each corner of the grid."
Grid: empty.
{"steps":[{"action":"place_at_corners","color":"Red","count":4}]}

═══════════════════════════════════════════════════════
OUTPUT: {"steps": [...]}  — JSON only. No text outside the JSON.
═══════════════════════════════════════════════════════
"""
