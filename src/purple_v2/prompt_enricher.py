"""Adaptive prompt enrichment: inject targeted concept reminders based on instruction keywords."""
from __future__ import annotations

import re
from typing import List


def enrich_prompt(instruction: str) -> str:
    """Return contextual enrichment text to append to the planner system prompt.

    Each rule fires only when the instruction matches a relevant pattern.
    Rules are short (<=5 lines) with concrete coordinate examples.
    """
    enrichments: List[str] = []
    lower = instruction.lower()

    # Rule 1: "in front of" = +z (NOT stacking)
    if re.search(r"\bin\s+front\s+of\b", lower):
        enrichments.append(
            "REMINDER: 'in front of' means +z direction (increasing z). "
            "If a block is at (0,0,0), 'in front of it' is (0,50,100). "
            "This is a HORIZONTAL move, NOT stacking on top."
        )

    # Rule 2: "behind" = -z
    if re.search(r"\bbehind\b", lower):
        enrichments.append(
            "REMINDER: 'behind' means -z direction (decreasing z). "
            "If a block is at (0,50,0), 'behind it' is (0,50,-100)."
        )

    # Rule 3: "each end" or "both ends" after extension
    if re.search(r"\b(each|both)\s+(end|side)s?\b", lower):
        enrichments.append(
            "REMINDER: 'each end'/'both ends' means the TWO endpoints of a row. "
            "After extending a row, RECALCULATE the endpoints! "
            "E.g., row at [-100,0,100] extended by 2 to the right -> new row is [-100,0,100,200,300]. "
            "New ends are x=-100 and x=300, NOT the old x=100."
        )

    # Rule 4: L-shape extension
    if re.search(r"\bl[\s-]?shape\b", lower) and re.search(r"\b(extend|add|longer|short)\b", lower):
        enrichments.append(
            "REMINDER: When extending an L-shape arm, extend AWAY from the junction/corner. "
            "NEVER stack on top. Use extend_row with direction pointing away from the corner."
        )

    # Rule 5: leftmost/rightmost
    if re.search(r"\b(leftmost|rightmost)\b", lower):
        enrichments.append(
            "REMINDER: 'leftmost' = block with MINIMUM x coordinate. "
            "'rightmost' = block with MAXIMUM x coordinate. "
            "'in front of the leftmost' = find min-x block, then place at same x but +z."
        )

    # Rule 6: corners
    if re.search(r"\bcorner\b", lower):
        enrichments.append(
            "REMINDER: Grid corners at ground level: "
            "bottom-left=(-400,50,400), bottom-right=(400,50,400), "
            "top-left=(-400,50,-400), top-right=(400,50,-400)."
        )

    # Rule 7: chain references ("the X one", "the X block you just placed")
    if re.search(r"\bthe\s+\w+\s+one\b", lower) or re.search(r"\byou\s+(just\s+)?placed\b", lower):
        enrichments.append(
            "REMINDER: Chain references like 'the green one' or 'the block you just placed' "
            "refer to the MOST RECENTLY placed block of that color, NOT the original position. "
            "Track where each color was LAST placed and use THAT position for subsequent steps."
        )

    # Rule 8: row extension
    if re.search(r"\b(extend|add\s+to|continue)\b", lower) and re.search(r"\b(row|line)\b", lower):
        enrichments.append(
            "REMINDER: When extending a row, start from the END of the existing row "
            "(the last block in the extension direction), not from the middle. "
            "Use extend_row with start position at the endpoint of the current row."
        )

    # Rule 9: "on top of" a specific block
    if re.search(r"\bon\s+top\s+of\b", lower):
        enrichments.append(
            "REMINDER: 'on top of' means same (x,z) position but higher y. "
            "Stack vertically: y increments by 100 for each block above."
        )

    # Rule 10: "starting from...going to" or "from...to"
    if re.search(r"\b(starting\s+from|from\s+the)\b.*\b(going|to\s+the)\b", lower):
        enrichments.append(
            "REMINDER: 'starting from X going to Y' defines a row from point X to point Y. "
            "Calculate the number of blocks: abs(end - start)/100 + 1. "
            "Direction is determined by which coordinate changes."
        )

    # Rule 11: "horizontal row" (not stacking)
    if re.search(r"\bhorizontal\b", lower) or (re.search(r"\brow\b", lower) and not re.search(r"\b(stack|tower)\b", lower)):
        enrichments.append(
            "REMINDER: A 'row' or 'horizontal' arrangement means blocks placed "
            "side by side at ground level (y=50), NOT stacked vertically."
        )

    # Rule 12: stack/tower
    if re.search(r"\b(stack|tower)\b", lower):
        enrichments.append(
            "REMINDER: A 'stack' or 'tower' is VERTICAL: blocks at same (x,z) with increasing y. "
            "A stack of 3 at (0,0): (0,50,0), (0,150,0), (0,250,0)."
        )

    # Rule 13: stack + directional placement
    if re.search(r"\b(stack|tower)\b", lower) and re.search(r"\b(to\s+the\s+(left|right)|in\s+front|behind)\b", lower):
        enrichments.append(
            "REMINDER: 'stack to the left of X' means build a vertical stack at a position "
            "that is horizontally displaced from X. First determine the horizontal position, "
            "then stack vertically there."
        )

    # Rule 14: T-shape
    if re.search(r"\bt[\s-]?shape\b", lower):
        enrichments.append(
            "REMINDER: T-shape anatomy: crossbar (horizontal line) + stem (perpendicular line). "
            "Junction is where stem meets crossbar (at an interior point, not endpoint). "
            "To extend the stem, continue in the same direction AWAY from the crossbar. "
            "NEVER stack on top of the stem. Use extend_row with the correct direction."
        )

    # Rule 15: "towards the bottom" = +z
    if re.search(r"\btowards?\s+the\s+bottom\b", lower):
        enrichments.append(
            "REMINDER: 'towards the bottom' of the grid means +z direction (increasing z). "
            "Bottom of grid is at z=400."
        )

    # Rule 16: "edge" placements
    if re.search(r"\b(left|right|top|bottom)\s+edge\b", lower) or re.search(r"\balong\s+(the\s+)?(left|right|top|bottom)\b", lower):
        enrichments.append(
            "REMINDER: Grid edges: left edge x=-400, right edge x=400, "
            "top edge z=-400, bottom edge z=400. "
            "'Along the left edge' means blocks at x=-400 with varying z."
        )

    # Rule 17: "middle" or "highlighted" square
    if re.search(r"\b(middle|highlighted|center)\s*(square|block|cell)?\b", lower):
        enrichments.append(
            "REMINDER: 'middle square'/'highlighted square'/'center' = origin (0,0). "
            "A row 'starting from the middle square going right' starts AT x=0, NOT x=100."
        )

    # Rule 18: "directly" placement
    if re.search(r"\bdirectly\s+(in\s+front|behind|to\s+the\s+(left|right)|above|below)\b", lower):
        enrichments.append(
            "REMINDER: 'directly in front/behind/left/right' means exactly ONE step (100 units) "
            "in that direction. No diagonal movement."
        )

    if not enrichments:
        return ""

    return "\n\n--- CONTEXTUAL REMINDERS ---\n" + "\n".join(enrichments)
