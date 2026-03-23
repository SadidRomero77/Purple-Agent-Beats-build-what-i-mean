"""Patch chain-reference coordinates by simulating step placement."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from .build_planner import BuildStep
from .grid import Grid, DIRECTION_OFFSETS

logger = logging.getLogger(__name__)


def patch_chain_references(steps: List[BuildStep], start_grid: Grid) -> List[BuildStep]:
    """Fix chain references by simulating where each color actually lands.

    Problem: LLM references "the green block" using the ORIGINAL position,
    but a previous step may have placed green at a NEW position.

    Solution: Simulate each step, track last_pos_by_color, and patch subsequent
    steps' positions when they reference a color that moved.
    """
    last_pos: Dict[str, Tuple[int, int]] = {}

    # Initialize from existing grid
    for b in start_grid.blocks:
        last_pos[b.color] = (b.x, b.z)

    for i, step in enumerate(steps):
        # Patch this step's reference if it mentions a color we've tracked
        if step.reference:
            _patch_reference_position(step, last_pos)

        # Simulate where this step places blocks
        try:
            end_pos = _simulate_step_end(step, last_pos)
            if end_pos:
                color = step.color.capitalize()
                last_pos[color] = end_pos
        except Exception:
            pass  # Don't break on simulation errors

    return steps


def _patch_reference_position(step: BuildStep, last_pos: Dict[str, Tuple[int, int]]) -> None:
    """Patch a step's position based on where its reference color actually is."""
    ref_lower = step.reference.lower()

    for color, (cx, cz) in last_pos.items():
        if color.lower() in ref_lower:
            # Check if the step's position references stale coordinates
            # by looking for coordinate patterns in the reference
            m = re.search(r'at[_\s]+(-?\d+)[_\s]+(-?\d+)', step.reference)
            if m:
                old_x, old_z = int(m.group(1)), int(m.group(2))
                if (old_x, old_z) != (cx, cz):
                    logger.info(
                        "Chain patch step ref '%s': (%d,%d) -> (%d,%d)",
                        step.reference, old_x, old_z, cx, cz,
                    )
                    # Update the reference string
                    step.reference = re.sub(
                        r'at[_\s]+(-?\d+)[_\s]+(-?\d+)',
                        f'at_{cx}_{cz}',
                        step.reference,
                    )
                    # Also update position if it matches old coords
                    if step.position.get("x") == old_x and step.position.get("z") == old_z:
                        step.position["x"] = cx
                        step.position["z"] = cz

            # For place_relative, update position based on reference + direction
            if step.action == "place_relative" and step.direction:
                offset = DIRECTION_OFFSETS.get(step.direction, (0, 0))
                step.position["x"] = cx + offset[0]
                step.position["z"] = cz + offset[1]

            break


def _simulate_step_end(step: BuildStep, last_pos: Dict[str, Tuple[int, int]]) -> Tuple[int, int] | None:
    """Simulate where a step's blocks end up, return the last position."""
    pos = step.position

    if step.action in ("stack", "place"):
        if "x" in pos and "z" in pos:
            return int(pos["x"]), int(pos["z"])

    elif step.action == "place_relative":
        if step.reference:
            ref_lower = step.reference.lower()
            for color, (cx, cz) in last_pos.items():
                if color.lower() in ref_lower:
                    offset = DIRECTION_OFFSETS.get(step.direction, (0, 0))
                    return cx + offset[0], cz + offset[1]
        if "x" in pos and "z" in pos:
            return int(pos["x"]), int(pos["z"])

    elif step.action == "extend_row":
        if "x" in pos and "z" in pos:
            x, z = int(pos["x"]), int(pos["z"])
            count = step.count if isinstance(step.count, int) else 3
            offset = DIRECTION_OFFSETS.get(step.direction, (100, 0))
            end_x = x + offset[0] * (count - 1)
            end_z = z + offset[1] * (count - 1)
            return end_x, end_z

    elif step.action == "place_at_corners":
        return 400, -400  # last corner placed

    return None
