"""
BuilderAgent - Core agent logic for the build_what_i_mean benchmark.

Uses LLM-powered spatial reasoning with pragmatic speaker modeling
to interpret building instructions and construct block structures.
"""

import json
import logging
import os
import re

import litellm

from agent.grid import (
    VALID_X, VALID_Z, VALID_Y, VALID_COLORS,
    parse_blocks, blocks_to_string, normalize_block_set,
)
from agent.instruction_parser import parse_message
from agent.speaker_model import SpeakerModel, SpeakerHistory

logger = logging.getLogger(__name__)

# LLM Configuration
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4o")
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.0"))

SYSTEM_PROMPT = """You are an expert block-building agent operating on a 9×9 grid.

## COORDINATE SYSTEM
- Grid: 9×9 cells in the x-z plane
- Origin (0,0,0) is the center, highlighted
- Valid X values: [-400, -300, -200, -100, 0, 100, 200, 300, 400]
- Valid Z values: [-400, -300, -200, -100, 0, 100, 200, 300, 400]
- Y values (height): 50 (ground), 150, 250, 350, 450 (each +100 per stacked block)
- Corners: bottom-left (-400,0,400), bottom-right (400,0,400), top-right (400,0,-400), top-left (-400,0,-400)
- "Front" is towards positive Z, "back" is towards negative Z
- "Left" is towards negative X, "right" is towards positive X

## GRID LAYOUT (top view, X increases left-to-right, Z increases top-to-bottom)
Z=-400: (-400,z) (-300,z) (-200,z) (-100,z) (0,z)   (100,z)  (200,z)  (300,z)  (400,z)    <- BACK ROW (top)
Z=-300: ...
Z=-200: ...
Z=-100: ...
Z=0:    ...                                    (0,0) CENTER
Z=100:  ...
Z=200:  ...
Z=300:  ...
Z=400:  (-400,z) (-300,z) (-200,z) (-100,z) (0,z)   (100,z)  (200,z)  (300,z)  (400,z)    <- FRONT ROW (bottom)

## AVAILABLE COLORS
Red, Blue, Green, Yellow, Purple (always capitalize)

## YOUR TASK
Given an instruction and optionally existing blocks on the grid, determine the COMPLETE final block configuration.

## CRITICAL RULES
1. Your output MUST include ALL blocks on the grid - both existing (start) blocks AND any new blocks you place
2. Use exact coordinate values from the valid lists above
3. Ground level is y=50; stack by adding +100 for each block above
4. Colors must be capitalized: Red, Blue, Green, Yellow, Purple
5. Think step-by-step about spatial positions before giving coordinates
6. When stacking, ensure blocks below exist first

## EXAMPLES

### Example 1 — Relative positioning
Start blocks: Blue,0,50,0
Instruction: "Add a red block to the right of the existing block"
Reasoning: "Right" = positive X direction. Blue block is at x=0, so one step right is x=100. Same row (z=0), ground level (y=50).
Output: {"reasoning": "Right of x=0 is x=100, same z=0, ground y=50", "blocks": [{"color": "Blue", "x": 0, "y": 50, "z": 0}, {"color": "Red", "x": 100, "y": 50, "z": 0}]}

### Example 2 — Stacking
Start blocks: Red,200,50,-100
Instruction: "Stack a green block on top of it"
Reasoning: "On top" = same x,z position, y increases by 100. Red is at y=50, so green goes at y=150.
Output: {"reasoning": "Same x=200, z=-100. On top of y=50 means y=150", "blocks": [{"color": "Red", "x": 200, "y": 50, "z": -100}, {"color": "Green", "x": 200, "y": 150, "z": -100}]}

### Example 3 — Ambiguous color (conservative interpretation)
Start blocks: Yellow,-100,50,0
Instruction: "Place two more blocks next to it"
Reasoning: Color not specified → use same color as existing block (Yellow, conservative). "Next to" = adjacent in x or z. Place at x=-200,z=0 and x=0,z=0.
Output: {"reasoning": "No color given, use Yellow. Adjacent to x=-100 are x=-200 and x=0", "blocks": [{"color": "Yellow", "x": -100, "y": 50, "z": 0}, {"color": "Yellow", "x": -200, "y": 50, "z": 0}, {"color": "Yellow", "x": 0, "y": 50, "z": 0}]}

## OUTPUT FORMAT
Respond with ONLY a JSON object:
{
  "reasoning": "your step-by-step spatial reasoning",
  "blocks": [
    {"color": "Red", "x": 0, "y": 50, "z": 0},
    ...
  ]
}

Include ALL blocks that should be on the grid after following the instruction."""


QUESTION_DECISION_PROMPT = """You are evaluating whether a building instruction is ambiguous enough to warrant asking a clarification question.

## CONTEXT
- Asking a question costs -5 points
- Building correctly earns +10, incorrectly costs -10
- So asking is worth it only if you'd likely guess wrong (>50% chance of error)

## SPEAKER CONTEXT
{speaker_context}

## INSTRUCTION
{instruction}

## EXISTING BLOCKS ON GRID
{start_blocks}

## YOUR TASK
Analyze:
1. Is the instruction fully specified (clear color, quantity, position)?
2. If ambiguous, what is ambiguous (color? number? position?)?
3. Given the speaker context, would you likely guess correctly?

Respond with ONLY a JSON object:
{
  "is_ambiguous": true/false,
  "ambiguity_type": "color_under" | "number_under" | "position_under" | "fully_spec",
  "confidence_without_asking": 0.0-1.0,
  "should_ask": true/false,
  "question": "the question to ask if should_ask is true"
}"""


class BuilderAgent:
    """
    Stateful agent that processes building instructions across a game session.
    Maintains speaker models and conversation history.
    """

    def __init__(self):
        self.speaker_model = SpeakerModel()
        self.conversation_history: list[dict] = []
        self.current_speaker: str | None = None
        self.current_instruction: str | None = None
        self.current_start_blocks: list[tuple[str, int, int, int]] = []
        self.current_ambiguity_type: str = "fully_spec"
        self.used_conservative: bool = False
        self.last_was_ask: bool = False
        self.pending_answer: str | None = None
        self.last_built_blocks: str = ""  # tracks last response for history
        self.round_count: int = 0
        self.game_count: int = 0

    async def process_message(self, text: str) -> str:
        """Process a message from the green agent and return a response."""
        parsed = parse_message(text)

        # Handle new game transition
        if parsed.is_new_game:
            self.game_count += 1
            self.speaker_model.reset()
            self.conversation_history.clear()
            self.round_count = 0
            logger.info(f"New game started (game #{self.game_count})")
            return "[BUILD];"  # Acknowledge - green agent will send the real task next

        # Handle feedback from previous round
        if parsed.is_feedback:
            self._process_feedback(parsed)
            return "[BUILD];"  # Acknowledge - green agent will send next instruction

        # Handle answer to our question
        if parsed.is_answer:
            self.pending_answer = parsed.answer_text
            logger.info(f"Received answer: {self.pending_answer}")
            return await self._build_with_context()

        # Handle invalid format feedback
        if "Invalid response format" in text:
            if self.current_instruction:
                return await self._build_with_context()
            return "[BUILD];"

        # New instruction
        if parsed.instruction_text:
            self.round_count += 1
            self.current_speaker = parsed.speaker_name
            self.current_instruction = parsed.instruction_text
            self.current_start_blocks = parsed.start_blocks
            self.pending_answer = None
            self.last_was_ask = False

            logger.info(
                f"Round {self.round_count}: Speaker={self.current_speaker}, "
                f"Instruction={parsed.instruction_text[:100]}"
            )

            # MEJORA #1: Always evaluate with LLM — no rigid heuristic
            ambiguity_type, question = await self._evaluate_instruction()
            self.current_ambiguity_type = ambiguity_type

            logger.info(f"LLM detected ambiguity: {ambiguity_type}")

            if question:
                self.last_was_ask = True
                return f"[ASK];{question}"

            return await self._build_with_context()

        # Fallback
        logger.warning(f"Unrecognized message format: {text[:200]}")
        return "[BUILD];"

    def _process_feedback(self, parsed):
        """Update speaker model and save round to conversation history."""
        if self.current_speaker and parsed.feedback_correct is not None:
            self.speaker_model.record_result(
                speaker_name=self.current_speaker,
                was_correct=parsed.feedback_correct,
                was_ambiguous=self.current_ambiguity_type != "fully_spec",
                used_conservative=self.used_conservative,
            )
            logger.info(
                f"Feedback for {self.current_speaker}: "
                f"correct={parsed.feedback_correct}, "
                f"score={parsed.feedback_score}, "
                f"total={parsed.feedback_total}"
            )

            # MEJORA #3: Append completed round to conversation history
            if self.current_instruction:
                self.conversation_history.append({
                    "round": self.round_count,
                    "speaker": self.current_speaker,
                    "instruction": self.current_instruction,
                    "built": self.last_built_blocks,
                    "correct": parsed.feedback_correct,
                    "score": parsed.feedback_score,
                })
                # Keep only last 5 rounds to avoid prompt bloat
                if len(self.conversation_history) > 5:
                    self.conversation_history.pop(0)

    def _build_speaker_context(self, speaker: SpeakerHistory) -> str:
        """Build rich speaker context string for LLM prompts."""
        if speaker.total_rounds == 0:
            return f"Speaker '{speaker.name}': No history yet. Assume standard cooperative communication."

        reliability = "unknown"
        if speaker.is_likely_reliable:
            reliability = "reliable/consistent (tends to use same color as existing blocks, literal quantities)"
        elif speaker.is_likely_unreliable:
            reliability = "unpredictable (mixes conservative and liberal interpretations)"

        context = (
            f"Speaker '{speaker.name}': {speaker.total_rounds} rounds seen, "
            f"{speaker.correct_rounds} correct ({speaker.accuracy:.0%} accuracy), "
            f"reliability: {reliability}."
        )

        if speaker.ambiguous_rounds > 0:
            context += (
                f" On {speaker.ambiguous_rounds} ambiguous instructions: "
                f"{speaker.ambiguous_correct_conservative} correct with conservative interpretation "
                f"(same color/fewer blocks), "
                f"{speaker.ambiguous_correct_liberal} correct with liberal interpretation "
                f"(different color/more blocks)."
            )

        return context

    async def _evaluate_instruction(self) -> tuple[str, str | None]:
        """
        MEJORA #1: Always evaluate instruction ambiguity with LLM.
        Replaces the rigid keyword heuristic that previously blocked this path.
        Returns (ambiguity_type, question_to_ask_or_None).
        """
        speaker = self.speaker_model.get_speaker(self.current_speaker or "unknown")
        speaker_context = self._build_speaker_context(speaker)
        start_str = blocks_to_string(self.current_start_blocks) if self.current_start_blocks else "(empty grid)"

        prompt = QUESTION_DECISION_PROMPT.format(
            speaker_context=speaker_context,
            instruction=self.current_instruction,
            start_blocks=start_str,
        )

        try:
            response = await litellm.acompletion(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You analyze building instructions for ambiguity."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=500,
            )

            result = json.loads(response.choices[0].message.content)
            ambiguity_type = result.get("ambiguity_type", "fully_spec")

            # Ask only if both LLM and speaker model agree it's worth the -5 cost
            llm_should_ask = result.get("should_ask", False) and bool(result.get("question", "").strip())
            speaker_allows_ask = self.speaker_model.should_ask_question(self.current_speaker, ambiguity_type)

            if llm_should_ask and speaker_allows_ask:
                return ambiguity_type, result["question"]

            return ambiguity_type, None

        except Exception as e:
            logger.error(f"Error evaluating instruction: {e}")
            return "fully_spec", None

    async def _build_with_context(self) -> str:
        """Build the structure using LLM reasoning."""
        if not self.current_instruction:
            return "[BUILD];"

        start_str = blocks_to_string(self.current_start_blocks) if self.current_start_blocks else "(no blocks)"

        # Speaker hint — no longer gated behind ambiguous_rounds >= 2
        speaker_hint = ""
        if self.current_speaker and self.current_ambiguity_type != "fully_spec":
            pref = self.speaker_model.get_interpretation_preference(
                self.current_speaker, self.current_ambiguity_type
            )
            speaker = self.speaker_model.get_speaker(self.current_speaker)
            speaker_context = self._build_speaker_context(speaker)

            if pref == "conservative":
                speaker_hint = (
                    f"\n\nSPEAKER CONTEXT ({self.current_ambiguity_type}): {speaker_context} "
                    f"Use conservative interpretation: same color as existing blocks, literal/smaller quantities."
                )
            else:
                speaker_hint = (
                    f"\n\nSPEAKER CONTEXT ({self.current_ambiguity_type}): {speaker_context} "
                    f"Use liberal interpretation: may mean a different color or more blocks than literally stated."
                )
            self.used_conservative = (pref == "conservative")
        else:
            self.used_conservative = True

        # Clarification answer context
        answer_context = ""
        if self.pending_answer:
            answer_context = (
                f"\n\nCLARIFICATION ANSWER: I asked a question and received: \"{self.pending_answer}\"\n"
                f"Use this information to resolve any ambiguity."
            )

        # MEJORA #3: Include recent rounds from conversation history
        history_context = ""
        if self.conversation_history:
            history_lines = []
            for entry in self.conversation_history[-3:]:
                result_str = "Correct" if entry["correct"] else "Incorrect"
                history_lines.append(
                    f"  Round {entry['round']} [{entry['speaker']}]: \"{entry['instruction']}\" -> {result_str}"
                )
            history_context = "\n\nPREVIOUS ROUNDS THIS GAME (use for spatial and speaker context):\n" + "\n".join(history_lines)

        user_prompt = f"""INSTRUCTION: {self.current_instruction}

EXISTING BLOCKS ON GRID (START_STRUCTURE): {start_str}{speaker_hint}{answer_context}{history_context}

Remember: Output ALL blocks that should be on the grid after following the instruction (including the start blocks if they should remain). Respond with ONLY the JSON object."""

        content = ""
        try:
            response = await litellm.acompletion(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            blocks = result.get("blocks", [])
            if not blocks:
                logger.warning("LLM returned no blocks")
                if self.current_start_blocks:
                    return "[BUILD];" + blocks_to_string(self.current_start_blocks)
                return "[BUILD];"

            # Convert to block tuples and validate
            validated_blocks = []
            for block in blocks:
                color = block.get("color", "").capitalize()
                x = int(block.get("x", 0))
                y = int(block.get("y", 50))
                z = int(block.get("z", 0))

                if color.upper() not in VALID_COLORS:
                    logger.warning(f"Invalid color: {color}")
                    continue
                if x not in VALID_X:
                    x = min(VALID_X, key=lambda v: abs(v - x))
                if z not in VALID_Z:
                    z = min(VALID_Z, key=lambda v: abs(v - z))
                if y not in VALID_Y:
                    y = min(VALID_Y, key=lambda v: abs(v - y))

                validated_blocks.append((color, x, y, z))

            # Remove duplicates preserving order
            seen = set()
            unique_blocks = []
            for block in validated_blocks:
                key = f"{block[0]},{block[1]},{block[2]},{block[3]}"
                if key not in seen:
                    seen.add(key)
                    unique_blocks.append(block)

            block_str = blocks_to_string(unique_blocks)
            self.last_built_blocks = block_str  # MEJORA #3: store for history
            return f"[BUILD];{block_str}"

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from LLM: {e}")
            return self._fallback_parse(content)

        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            if self.current_start_blocks:
                return "[BUILD];" + blocks_to_string(self.current_start_blocks)
            return "[BUILD];"

    def _fallback_parse(self, content: str) -> str:
        """Try to extract block coordinates from a non-JSON LLM response."""
        pattern = r'(Red|Blue|Green|Yellow|Purple),\s*(-?\d+),\s*(\d+),\s*(-?\d+)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            blocks = []
            for color, x, y, z in matches:
                blocks.append((color.capitalize(), int(x), int(y), int(z)))
            return f"[BUILD];{blocks_to_string(blocks)}"

        if self.current_start_blocks:
            return "[BUILD];" + blocks_to_string(self.current_start_blocks)
        return "[BUILD];"
