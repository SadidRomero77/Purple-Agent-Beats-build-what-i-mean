"""LLM-based ambiguity detection via BAML.

Replaces the broken regex heuristics from v2. Uses GPT-4o-mini to detect
whether an instruction is missing a color or count for any block placement.

Strategy: Ask whenever ANY ambiguity is detected (-5 cost but +10 if answer
helps build correctly, vs -10 for wrong guess).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Set

from .grid import Grid

logger = logging.getLogger(__name__)

KNOWN_COLORS: Set[str] = {
    "red", "blue", "green", "yellow", "purple",
    "orange", "white", "black", "brown", "pink",
    "grey", "gray", "cyan",
}


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


async def detect_ambiguity_llm(
    instruction: str,
    start_grid: Grid,
) -> AmbiguityInfo:
    """Detect ambiguity using BAML LLM call."""
    try:
        from .baml_client.baml_client import b

        # Gather context colors from the grid
        context_colors = ", ".join(sorted(start_grid.unique_colors())) if start_grid.blocks else "none"

        result = await b.DetectAmbiguity(
            instruction=instruction,
            start_structure=start_grid.to_str() or "(empty)",
            context_colors=context_colors,
        )

        info = AmbiguityInfo(
            has_missing_color=result.has_missing_color,
            has_missing_count=result.has_missing_count,
            suggested_color_question=result.suggested_color_question or "What color should the unspecified blocks be?",
            suggested_count_question=result.suggested_count_question or "How many blocks should be in the stack?",
            inferred_color=result.inferred_color or _infer_color_from_context(instruction, start_grid),
            inferred_count=result.inferred_count if result.inferred_count > 0 else 3,
            reasoning=result.reasoning,
        )

        logger.info("BAML ambiguity: color=%s count=%s reason=%s",
                     info.has_missing_color, info.has_missing_count, info.reasoning[:100])
        return info

    except Exception as exc:
        logger.warning("BAML ambiguity detection failed, falling back to heuristic: %s", exc)
        return _detect_ambiguity_heuristic(instruction, start_grid)


def _infer_color_from_context(instruction: str, grid: Grid) -> str:
    """Infer the most likely color for unspecified blocks.

    Priority order:
    1. If instruction has exactly one color → use that color
    2. If instruction has multiple colors → use the LAST mentioned color
       (the colorless phrase usually follows the last colored one)
    3. If grid has blocks → use the most common color on the grid
    4. If grid has blocks from instruction colors → use the one closest
       to the colorless phrase position
    5. Last resort → use "Red" (most common in stimulus data, never "Purple")
    """
    lower = instruction.lower()

    # Find colors mentioned in instruction, preserving order of appearance
    instruction_colors = []
    for m in re.finditer(
        r'\b(' + '|'.join(KNOWN_COLORS) + r')\b', lower
    ):
        c = m.group(1).capitalize()
        if c not in instruction_colors:
            instruction_colors.append(c)

    if len(instruction_colors) == 1:
        return instruction_colors[0]

    if len(instruction_colors) > 1:
        # Use the last mentioned color — colorless phrases typically follow
        # the last colored clause and refer to the same context
        return instruction_colors[-1]

    # No colors in instruction — use grid context
    if grid.blocks:
        return grid.most_common_color()

    # Absolute last resort — "Red" is the most common color in the game,
    # never default to "Purple" which is rarely the intended answer
    return "Red"


def _detect_ambiguity_heuristic(instruction: str, grid: Grid) -> AmbiguityInfo:
    """Fallback heuristic detection when BAML is unavailable."""
    info = AmbiguityInfo()
    lower = instruction.lower()

    # Find all color mentions
    instruction_colors = set()
    for color in KNOWN_COLORS:
        if re.search(r'\b' + color + r'\b', lower):
            instruction_colors.add(color)

    # Check for block-placing phrases without a color
    # Patterns allow optional numbers/words between verb and noun
    _NUM_WORDS = r'(?:\d+\s+|a\s+|an\s+|the\s+|some\s+|one\s+|two\s+|three\s+|four\s+|five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)*'
    placing_patterns = [
        r'\bstack\s+' + _NUM_WORDS + r'blocks?\b',
        r'\bbuild\s+(?:a\s+)?' + _NUM_WORDS + r'(?:stack|tower)\b',
        r'\bplace\s+' + _NUM_WORDS + r'(?:blocks?|horizontal\s+row)\b',
        r'\badd\s+' + _NUM_WORDS + r'blocks?\b',
        r'\bput\s+' + _NUM_WORDS + r'blocks?\b',
    ]

    # Split by clauses
    clauses = re.split(r'\.\s*|\bthen\b|\band\s+then\b', lower)

    for clause in clauses:
        has_placing = any(re.search(p, clause) for p in placing_patterns)
        if not has_placing:
            continue

        has_color = any(re.search(r'\b' + c + r'\b', clause) for c in KNOWN_COLORS)
        if not has_color:
            info.has_missing_color = True
            info.suggested_color_question = "What color should the unspecified blocks be?"

    # Check for missing count
    count_patterns = [
        r'\bstack\s+(?:' + '|'.join(KNOWN_COLORS) + r')?\s*blocks?\b',
        r'\bbuild\s+(?:a\s+)?(?:' + '|'.join(KNOWN_COLORS) + r')?\s*(?:stack|tower)\b',
    ]

    for clause in clauses:
        for pattern in count_patterns:
            m = re.search(pattern, clause)
            if not m:
                continue
            # Check for number in surrounding context
            context = clause
            has_number = bool(re.search(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a)\b', context))
            if not has_number:
                info.has_missing_count = True
                info.suggested_count_question = "How many blocks should be in the stack?"

    info.inferred_color = _infer_color_from_context(instruction, grid)
    info.inferred_count = 3

    return info


def patch_instruction_with_color(instruction: str, color: str) -> str:
    """Patch colorless block-placing phrases with the given color."""
    _NUM_WORDS_PAT = r'(?:\d+\s+|a\s+|an\s+|the\s+|some\s+|one\s+|two\s+|three\s+|four\s+|five\s+|six\s+|seven\s+|eight\s+|nine\s+|ten\s+)*'

    def _insert_color(match):
        pre = match.group(0)
        for c in KNOWN_COLORS:
            if c in pre.lower():
                return pre
        noun_match = re.search(r'\b(blocks?|stack|tower|row)\b', pre, re.IGNORECASE)
        if noun_match:
            idx = noun_match.start()
            return pre[:idx] + color.lower() + " " + pre[idx:]
        return pre

    patterns = [
        r'\b(stack|place|build|put|add)\s+' + _NUM_WORDS_PAT + r'(?:blocks?|stack|tower)',
        r'\b(extend|continue)\s+(?:the\s+)?(?:row|line)',
    ]

    result = instruction
    for pattern in patterns:
        result = re.sub(pattern, _insert_color, result, flags=re.IGNORECASE)

    return result


def patch_instruction_with_count(
    instruction: str, count: int, target_color: str = ""
) -> str:
    """Patch missing count into the instruction."""
    lower = instruction.lower()

    # Patterns that indicate a missing count, with their replacements
    _COLOR_PAT = r'(?:(?:' + '|'.join(KNOWN_COLORS) + r')\s+)?'
    patterns = [
        (r'\bstack\s+' + _COLOR_PAT + r'(blocks)\b', lambda m: m.group(0).replace(m.group(1) if m.lastindex else 'blocks', f'{count} blocks')),
        (r'\bbuild\s+a\s+' + _COLOR_PAT + r'(stack|tower)\b', lambda m: m.group(0).replace(m.group(0).split()[-1], f'stack of {count}')),
        (r'\bplace\s+(blocks)\b', lambda m: f'place {count} blocks'),
        (r'\badd\s+(blocks)\b', lambda m: f'add {count} blocks'),
    ]

    # Simpler approach: insert count before "stack/tower" when no number present
    # Check if there's already a number near "stack"/"tower"
    if not re.search(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:blocks?|stack|tower)\b', lower):
        # Insert "of {count}" after "stack" or "tower" when it's a noun
        result = re.sub(
            r'\b((?:a|the)\s+(?:\w+\s+)?(?:stack|tower))\b',
            lambda m: m.group(0) + f' of {count}',
            instruction,
            count=1,
            flags=re.IGNORECASE,
        )
        if result != instruction:
            return result

        # Insert count before "blocks"
        result = re.sub(
            r'\b(stack|place|build|add)\s+(blocks)\b',
            lambda m: f'{m.group(1)} {count} blocks',
            instruction,
            count=1,
            flags=re.IGNORECASE,
        )
        if result != instruction:
            return result

    return instruction
