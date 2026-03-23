"""Precise heuristic ambiguity detection for Build What I Mean.

Replaces the BAML LLM-based detector (v3) with calibrated regex heuristics.
Key improvement: checks if color/count exists BETWEEN the placing verb and
the block noun, not just anywhere in the clause. This eliminates false
positives like "stack blocks behind the RED one" (red is a reference, not
the placed block's color).

Strategy: EV-based ASK decisions. Ask only when EV(ask) > EV(guess).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .grid import Grid

logger = logging.getLogger(__name__)

KNOWN_COLORS: Set[str] = {
    "red", "blue", "green", "yellow", "purple",
    "orange", "white", "black", "brown", "pink",
    "grey", "gray", "cyan",
}

_COLOR_PATTERN = '|'.join(KNOWN_COLORS)

_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_NUMBER_WORDS = '|'.join(_WORD_TO_INT.keys())

# Verbs that imply placing blocks
_PLACING_VERBS = r'(?:stack|place|build|add|put|extend|create|make)'
# Nouns that represent block objects
_BLOCK_NOUNS = r'(?:blocks?|stack|tower|row|line|column)'


@dataclass
class AmbiguityInfo:
    """Result of ambiguity detection."""
    has_missing_color: bool = False
    has_missing_count: bool = False
    suggested_color_question: str = ""
    suggested_count_question: str = ""
    inferred_color: str = ""
    inferred_count: int = 3
    reasoning: str = ""


def detect_ambiguity(instruction: str, grid: Grid) -> AmbiguityInfo:
    """Detect missing color and/or count using precise regex heuristics.

    Key logic: checks if color/count exists BETWEEN the placing verb and
    the block noun in each clause. "stack blocks behind the red one" has
    no color between "stack" and "blocks" → missing color.
    """
    info = AmbiguityInfo()
    lower = instruction.lower()

    # Split into clauses
    clauses = re.split(r'\.\s+|\bthen\b|\band\s+then\b|,\s*then\b|;\s*', lower)

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # Find placing verb → block noun patterns
        for m in re.finditer(
            r'\b' + _PLACING_VERBS + r'\b(.*?)\b' + _BLOCK_NOUNS + r'\b',
            clause,
        ):
            between = m.group(1)  # text BETWEEN verb and noun

            # Check for color BETWEEN verb and noun
            has_color_between = bool(re.search(
                r'\b(' + _COLOR_PATTERN + r')\b', between
            ))

            # Check for count BETWEEN verb and noun
            has_count_between = bool(re.search(
                r'\b(\d+|' + _NUMBER_WORDS + r')\b', between
            ))

            # "a" or "an" before noun implies count=1 (not missing)
            has_article = bool(re.search(r'\b(a|an)\b', between))

            if not has_color_between:
                # Verify this is actually a placing phrase (has a block noun)
                if re.search(r'\b' + _BLOCK_NOUNS + r'\b', clause):
                    info.has_missing_color = True

            if not has_count_between and not has_article:
                # "Build a stack" → "a" implies 1, not missing
                # "Stack blocks" → genuinely missing count
                # But "Build a yellow stack" → "a" is article for "stack", count IS missing
                # Check: is there a "a/an" right before the noun (not between verb and color)?
                full_match = m.group(0)
                if not re.search(r'\ba\s+' + _BLOCK_NOUNS + r'\b', full_match):
                    info.has_missing_count = True
                elif re.search(r'\ba\s+(?:' + _COLOR_PATTERN + r'\s+)?' + r'(?:stack|tower)\b', full_match):
                    # "Build a yellow stack" → count IS missing (how tall?)
                    info.has_missing_count = True

        # Also check for "Build a [color] stack/tower" without "of N"
        stack_match = re.search(
            r'\b(?:build|make|create)\s+a\s+(?:(?:' + _COLOR_PATTERN + r')\s+)?(?:stack|tower)\b',
            clause,
        )
        if stack_match:
            # Check if count follows ("of 3", "of three")
            after_match = clause[stack_match.end():]
            has_count_after = bool(re.search(
                r'^\s*(?:of\s+)?(\d+|' + _NUMBER_WORDS + r')\b', after_match
            ))
            if not has_count_after:
                info.has_missing_count = True

    # Generate suggested questions
    if info.has_missing_color and info.has_missing_count:
        info.suggested_color_question = (
            "What color should the unspecified blocks be, "
            "and how many blocks should there be?"
        )
        info.suggested_count_question = info.suggested_color_question
    elif info.has_missing_color:
        # Include context about what colors exist
        instruction_colors = _extract_instruction_colors(lower)
        if instruction_colors:
            colors_str = ", ".join(c.capitalize() for c in sorted(instruction_colors))
            info.suggested_color_question = (
                f"What color should the unspecified blocks be? "
                f"The instruction mentions {colors_str}."
            )
        else:
            info.suggested_color_question = "What color should the unspecified blocks be?"
    elif info.has_missing_count:
        info.suggested_count_question = "How many blocks should be in the stack?"

    # Infer defaults
    info.inferred_color = infer_color_from_context(instruction, grid)
    info.inferred_count = infer_count_from_context(instruction, grid)
    info.reasoning = _build_reasoning(info, lower)

    return info


def _extract_instruction_colors(lower: str) -> Set[str]:
    """Extract all color names mentioned in instruction text."""
    colors = set()
    for color in KNOWN_COLORS:
        if re.search(r'\b' + color + r'\b', lower):
            colors.add(color)
    return colors


def infer_color_from_context(instruction: str, grid: Grid) -> str:
    """Infer the most likely color for unspecified blocks.

    Priority:
    1. Single color in instruction → use that
    2. Multiple colors → use the LAST mentioned
    3. Grid context → most common grid color
    4. Last resort → "Red"
    """
    lower = instruction.lower()

    # Find colors in order of appearance
    instruction_colors = []
    for m in re.finditer(r'\b(' + _COLOR_PATTERN + r')\b', lower):
        c = m.group(1).capitalize()
        if c not in instruction_colors:
            instruction_colors.append(c)

    if len(instruction_colors) == 1:
        return instruction_colors[0]

    if len(instruction_colors) > 1:
        return instruction_colors[-1]

    if grid.blocks:
        return grid.most_common_color()

    return "Red"


def infer_count_from_context(instruction: str, grid: Grid) -> int:
    """Infer missing count from grid context.

    Priority:
    1. If instruction references a specific color stack, match its height
    2. If grid has stacks, use the most common stack height
    3. Default to 3
    """
    lower = instruction.lower()

    # Check if instruction references a specific stack color
    for color in KNOWN_COLORS:
        if re.search(r'\b' + color + r'\b', lower):
            color_blocks = grid.find_color(color.capitalize())
            if color_blocks:
                # Find the tallest stack of this color
                positions: Dict[Tuple[int, int], int] = {}
                for b in color_blocks:
                    key = (b.x, b.z)
                    positions[key] = positions.get(key, 0) + 1
                if positions:
                    return max(positions.values())

    # Use most common stack height on grid
    if grid.blocks:
        heights: Dict[Tuple[int, int], int] = {}
        for b in grid.blocks:
            key = (b.x, b.z)
            heights[key] = heights.get(key, 0) + 1
        if heights:
            # Return most common height
            height_counts: Dict[int, int] = {}
            for h in heights.values():
                height_counts[h] = height_counts.get(h, 0) + 1
            return max(height_counts, key=height_counts.get)  # type: ignore

    return 3


def _build_reasoning(info: AmbiguityInfo, lower: str) -> str:
    """Build a human-readable reasoning string."""
    parts = []
    if info.has_missing_color:
        parts.append("Missing color: placing verb found without color between verb and noun")
    if info.has_missing_count:
        parts.append("Missing count: no number/word-number between verb and noun")
    if not parts:
        parts.append("Fully specified: all placing phrases have color and count")
    return "; ".join(parts)


def patch_instruction_with_color(instruction: str, color: str) -> str:
    """Insert color into colorless block-placing phrases."""

    def _insert_color(match):
        pre = match.group(0)
        # Don't double-insert
        for c in KNOWN_COLORS:
            if c in pre.lower():
                return pre
        # Find noun and insert color before it
        noun_match = re.search(r'\b(' + _BLOCK_NOUNS + r')\b', pre, re.IGNORECASE)
        if noun_match:
            idx = noun_match.start()
            return pre[:idx] + color.lower() + " " + pre[idx:]
        return pre

    _NUM = r'(?:\d+\s+|a\s+|an\s+|the\s+|some\s+|one\s+|two\s+|three\s+|four\s+|five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)*'
    patterns = [
        r'\b' + _PLACING_VERBS + r'\s+' + _NUM + r'(?:' + _BLOCK_NOUNS + r')',
        r'\b(?:extend|continue)\s+(?:the\s+)?(?:row|line)',
    ]

    result = instruction
    for pattern in patterns:
        result = re.sub(pattern, _insert_color, result, flags=re.IGNORECASE)
    return result


def patch_instruction_with_count(
    instruction: str, count: int, target_color: str = ""
) -> str:
    """Insert count into countless block-placing phrases."""
    lower = instruction.lower()

    # Already has a number → don't patch
    if re.search(r'\b(\d+|' + _NUMBER_WORDS + r')\s+(?:' + _BLOCK_NOUNS + r')\b', lower):
        return instruction

    # "Build a [color] stack/tower" → "Build a [color] stack of N"
    result = re.sub(
        r'\b((?:a|the)\s+(?:\w+\s+)?(?:stack|tower))\b',
        lambda m: m.group(0) + f' of {count}',
        instruction,
        count=1,
        flags=re.IGNORECASE,
    )
    if result != instruction:
        return result

    # "stack [color] blocks" → "stack N [color] blocks"
    result = re.sub(
        r'\b(stack|place|build|add)\s+((?:' + _COLOR_PATTERN + r'\s+)?blocks)\b',
        lambda m: f'{m.group(1)} {count} {m.group(2)}',
        instruction,
        count=1,
        flags=re.IGNORECASE,
    )
    if result != instruction:
        return result

    return instruction
