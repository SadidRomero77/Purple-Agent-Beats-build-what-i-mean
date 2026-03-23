"""Detect underspecified instructions (missing color or count) using heuristics.

Strategy:
- Ask costs -5, correct answer earns +10, wrong answer costs -10.
- If color is missing: EV(ask) = -5 + 10 = +5; EV(guess) ~ 0 → ASK
- If count is missing: EV(ask) = -5 + 10 = +5; EV(guess) ~ +2.9 (64.6% heuristic) → ASK
- Only one question per round; prioritize color > count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


KNOWN_COLORS: Set[str] = {
    "red", "blue", "green", "yellow", "purple",
    "orange", "white", "black", "brown", "pink",
    "grey", "gray", "cyan",
}

# Phrases that imply placing blocks
_PLACING_PHRASES = [
    r"\bstack\b", r"\bplace\b", r"\bbuild\b", r"\badd\b",
    r"\bput\b", r"\bextend\b", r"\bmake\b", r"\bcreate\b",
]

# Number words
WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1,
}


@dataclass
class UnderspecResult:
    """Result of underspec detection."""
    has_missing_color: bool = False
    has_missing_number: bool = False
    inferred_color: str = ""
    inferred_count: int | None = None
    uncounted_color: str = ""
    suggested_question: str = ""
    suggested_count_question: str = ""
    suggested_compound_question: str = ""
    details: str = ""


def detect_underspec_heuristic(instruction: str) -> UnderspecResult:
    """Detect missing color and/or count using regex heuristics."""
    result = UnderspecResult()
    lower = instruction.lower()

    # Find all color mentions in the instruction
    instruction_colors = set()
    for color in KNOWN_COLORS:
        if re.search(r'\b' + color + r'\b', lower):
            instruction_colors.add(color)

    # Detect block-placing phrases that lack a color
    colorless_phrases = _find_colorless_phrases(lower, instruction_colors)

    if colorless_phrases:
        # Color is missing for some placed blocks
        if len(instruction_colors) == 0:
            # No colors mentioned at all — definitely need to ask
            result.has_missing_color = True
            result.suggested_question = "What color should the blocks be?"
            result.details = "No colors mentioned in instruction."
        elif len(instruction_colors) == 1:
            # One color + colorless phrase: might be same color or different
            only_color = list(instruction_colors)[0]
            # Heuristic: if instruction has multiple clauses, colorless clause may differ
            if _has_multiple_clauses(lower):
                result.has_missing_color = True
                result.inferred_color = only_color.capitalize()
                result.suggested_question = (
                    f"What color should the unspecified blocks be? "
                    f"Should they be {only_color.capitalize()} or a different color?"
                )
                result.details = f"Single color {only_color} with colorless phrases in multi-clause instruction."
            else:
                # Single clause, single color — likely same color, don't ask
                result.inferred_color = only_color.capitalize()
                result.details = f"Inferred same color: {only_color}."
        else:
            # Multiple colors + colorless phrase — must ask
            result.has_missing_color = True
            colors_str = ", ".join(c.capitalize() for c in sorted(instruction_colors))
            result.suggested_question = (
                f"What color should the unspecified blocks be? "
                f"The instruction mentions {colors_str}."
            )
            result.details = f"Multiple colors {instruction_colors} with colorless phrases."

    # Detect missing count
    _detect_missing_count(lower, instruction_colors, result)

    # Build compound question if both missing
    if result.has_missing_color and result.has_missing_number:
        result.suggested_compound_question = (
            "What color should the unspecified blocks be, "
            "and how many blocks should be in that stack?"
        )

    return result


def _find_colorless_phrases(lower: str, known_colors: Set[str]) -> List[str]:
    """Find phrases that imply placing blocks but don't specify a color."""
    colorless = []

    # Split into sub-instructions by "then", "and", comma, period
    parts = re.split(r'\b(then|and then|,\s*then|\.)\b', lower)

    for part in parts:
        has_placing = any(re.search(p, part) for p in _PLACING_PHRASES)
        if not has_placing:
            continue

        # Check if the PLACING verb has a direct color (not just a reference color)
        # "stack red blocks" → color before noun = specified
        # "stack blocks behind the blue one" → color only in reference = UNspecified
        placing_color_found = False
        for p in _PLACING_PHRASES:
            m = re.search(p, part)
            if m:
                # Look for color between the verb and the block noun
                after_verb = part[m.end():]
                # Color directly modifying the placed object (before "block/stack/tower")
                noun_m = re.search(r'\b(blocks?|stack|tower|row|line)\b', after_verb)
                if noun_m:
                    between = after_verb[:noun_m.start()]
                    for c in known_colors:
                        if re.search(r'\b' + c + r'\b', between):
                            placing_color_found = True
                            break
                else:
                    # No noun after verb — check if any color immediately follows
                    for c in known_colors:
                        if re.search(r'\b' + c + r'\b', after_verb[:30]):
                            placing_color_found = True
                            break
                if placing_color_found:
                    break

        if not placing_color_found:
            if re.search(r'\b(blocks?|stack|tower|row|line)\b', part):
                colorless.append(part.strip())

    return colorless


def _has_multiple_clauses(lower: str) -> bool:
    """Check if instruction has multiple clauses (then, and, comma)."""
    return bool(re.search(r'\b(then|and\s+then|afterwards?)\b', lower) or lower.count(',') > 1)


def _detect_missing_count(lower: str, colors: Set[str], result: UnderspecResult):
    """Detect if a block count is missing from the instruction."""
    # Look for patterns like "stack blocks" without a number
    count_patterns = [
        (r'\bstack\s+(?:the\s+)?(?:' + '|'.join(KNOWN_COLORS) + r')?\s*blocks?\b', True),
        (r'\bstack\s+blocks?\b', True),
        (r'\bbuild\s+a\s+(?:' + '|'.join(KNOWN_COLORS) + r')?\s*(?:stack|tower)\b', True),
        (r'\bplace\s+(?:some|several|more)\s+', True),
    ]

    for pattern, is_missing in count_patterns:
        m = re.search(pattern, lower)
        if m:
            phrase = m.group(0)
            # Check if there's a number before or in the phrase
            # Look back 30 chars for a number
            start = max(0, m.start() - 30)
            context = lower[start:m.end() + 30]
            has_number = bool(re.search(r'\b(\d+|' + '|'.join(WORD_TO_INT.keys()) + r')\b', context))
            if not has_number and is_missing:
                result.has_missing_number = True
                # Try to identify which color's count is missing
                for color in colors:
                    if color in phrase:
                        result.uncounted_color = color.capitalize()
                        break
                if not result.uncounted_color and colors:
                    result.uncounted_color = list(colors)[0].capitalize()
                result.suggested_count_question = (
                    f"How many blocks should be in the "
                    f"{result.uncounted_color.lower() + ' ' if result.uncounted_color else ''}stack?"
                )
                result.details += f" Missing count for '{phrase}'."
                break

    # Infer count from context if not asking
    if result.has_missing_number and not result.has_missing_color:
        # Try to find a nearby number that could apply
        numbers = re.findall(r'\b(\d+)\b', lower)
        for word, val in WORD_TO_INT.items():
            if word in ("a", "an"):
                continue
            if re.search(r'\b' + word + r'\b', lower):
                numbers.append(str(val))
        if numbers:
            result.inferred_count = int(numbers[-1])
        else:
            result.inferred_count = 3  # safe default


def patch_instruction_with_color(instruction: str, color: str) -> str:
    """Patch colorless block-placing phrases with the given color."""
    # Insert color before "block(s)", "stack", "tower" when no color precedes
    def _insert_color(match):
        pre = match.group(0)
        # Check if a color already precedes
        for c in KNOWN_COLORS:
            if c in pre.lower():
                return pre
        # Insert color before the noun
        noun_match = re.search(r'\b(blocks?|stack|tower|row)\b', pre, re.IGNORECASE)
        if noun_match:
            idx = noun_match.start()
            return pre[:idx] + color.lower() + " " + pre[idx:]
        return pre

    patterns = [
        r'\b(stack|place|build|put|add)\s+(?:a\s+|the\s+)?(?:blocks?|stack|tower)',
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

    # Try to insert count before "blocks" or "stack"
    patterns = [
        (r'\bstack\s+(blocks)\b', f'stack {count} blocks'),
        (r'\bbuild\s+a\s+(stack|tower)\b', f'build a stack of {count}'),
        (r'\bplace\s+(blocks)\b', f'place {count} blocks'),
        (r'\badd\s+(blocks)\b', f'add {count} blocks'),
    ]

    result = instruction
    for pattern, replacement in patterns:
        if re.search(pattern, lower):
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            return result

    return result
