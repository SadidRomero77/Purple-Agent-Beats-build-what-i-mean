"""Post-plan verification and deterministic auto-fixes."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .build_planner import BuildStep
from .grid import Grid, DIRECTION_OFFSETS
from .structure_analyzer import analyze_structure

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of plan verification."""
    issues: List[str] = field(default_factory=list)
    has_critical: bool = False

    def correction_prompt(self) -> str:
        if not self.issues:
            return ""
        parts = ["The previous plan had these issues:"]
        for issue in self.issues:
            parts.append(f"- FIX: {issue}")
        return "\n".join(parts)


def verify_plan(
    instruction: str, steps: List[BuildStep], start_block_count: int
) -> VerificationResult:
    """Verify the plan against the instruction text."""
    result = VerificationResult()
    lower = instruction.lower()

    # Check 1: Direction consistency
    _check_direction_consistency(lower, steps, result)

    # Check 2: "each/every" expansion
    _check_each_every(lower, steps, result)

    # Check 3: Stacking vs horizontal
    _check_stacking_vs_horizontal(lower, steps, result)

    # Check 4: Block count sanity
    _check_block_count(lower, steps, result)

    # Check 5: Ground-level placement when no stacking intended
    _check_ground_level(lower, steps, result)

    return result


def _check_direction_consistency(lower: str, steps: List[BuildStep], result: VerificationResult):
    """Check if extend_row directions match instruction text."""
    for step in steps:
        if step.action != "extend_row" or not step.direction:
            continue

        # Only flag if instruction has exactly one direction word
        has_left = bool(re.search(r'\b(left|going\s+left)\b', lower))
        has_right = bool(re.search(r'\b(right|going\s+right)\b', lower))

        if has_left and not has_right and step.direction == "right":
            result.issues.append(
                f"Instruction says 'left' but extend_row direction is 'right'. Fix direction to 'left'."
            )
            result.has_critical = True
        elif has_right and not has_left and step.direction == "left":
            result.issues.append(
                f"Instruction says 'right' but extend_row direction is 'left'. Fix direction to 'right'."
            )
            result.has_critical = True


def _check_each_every(lower: str, steps: List[BuildStep], result: VerificationResult):
    """Check if 'each'/'every' phrases are expanded into multiple steps."""
    if re.search(r'\b(each|every)\s+(end|corner|side|arm)\b', lower):
        # Should have at least 2 stack/place steps for endpoints
        stack_place_count = sum(
            1 for s in steps if s.action in ("stack", "place") and s.action != "extend_row"
        )
        if stack_place_count < 2:
            result.issues.append(
                "'each end/corner/arm' requires at least 2 separate stack/place steps. "
                "Make sure you place blocks at ALL endpoints, not just one."
            )
            result.has_critical = True


def _check_stacking_vs_horizontal(lower: str, steps: List[BuildStep], result: VerificationResult):
    """Check for inappropriate stacking when horizontal placement is intended."""
    horizontal_phrases = [
        r'\bin\s+front\s+of\b', r'\bbehind\b',
        r'\bto\s+the\s+(left|right)\b', r'\bhorizontal\b',
        r'\brow\b', r'\bline\b',
    ]
    is_horizontal = any(re.search(p, lower) for p in horizontal_phrases)
    has_stack_word = bool(re.search(r'\b(stack|tower|on\s+top)\b', lower))

    if is_horizontal and not has_stack_word:
        for step in steps:
            if step.action == "stack" and (isinstance(step.count, int) and step.count > 1):
                if not re.search(r'\b(stack|tower)\b', lower):
                    result.issues.append(
                        f"Instruction implies horizontal placement but plan uses 'stack' "
                        f"with count={step.count}. Use 'extend_row' or multiple 'place' steps instead."
                    )
                    result.has_critical = True


def _check_block_count(lower: str, steps: List[BuildStep], result: VerificationResult):
    """Sanity check on total block count."""
    # Extract numbers from instruction
    numbers = [int(n) for n in re.findall(r'\b(\d+)\b', lower)]
    word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    for word, val in word_nums.items():
        if re.search(r'\b' + word + r'\b', lower):
            numbers.append(val)

    if not numbers:
        return

    total_planned = 0
    for step in steps:
        if isinstance(step.count, int):
            total_planned += step.count
        else:
            total_planned += 3  # estimate for uncounted

    max_expected = sum(numbers) * 3  # generous upper bound
    if total_planned > max_expected and total_planned > 20:
        result.issues.append(
            f"Plan produces ~{total_planned} blocks but instruction mentions {numbers}. "
            f"This seems excessive. Double-check counts."
        )


def _check_ground_level(lower: str, steps: List[BuildStep], result: VerificationResult):
    """Check that non-stacking placements produce ground-level blocks.

    If the instruction doesn't say 'stack', 'on top', or 'tower' for a
    particular placement, the blocks should be at ground level (y=50).
    This catches the bug where caps on T-shape crossbar ends get stacked
    at y=150 instead of placed at y=50.
    """
    has_stack_context = bool(re.search(r'\b(stack|tower|on\s+top)\b', lower))

    # If the instruction is explicitly about stacking, don't flag
    if has_stack_context:
        return

    # Check for place/extend_row steps that might accidentally stack
    for step in steps:
        if step.action in ("place", "place_relative") and isinstance(step.count, int) and step.count > 1:
            result.issues.append(
                f"Step places {step.count} blocks with action='{step.action}' "
                f"but instruction doesn't mention stacking. Use action='extend_row' "
                f"for horizontal placement or action='stack' if vertical."
            )
            result.has_critical = True


# ─── Auto-fix functions ───


def auto_fix_direction(instruction: str, steps: List[BuildStep]) -> List[BuildStep]:
    """Fix extend_row direction if it contradicts instruction text."""
    lower = instruction.lower()

    has_left = bool(re.search(r'\b(going\s+)?left\b', lower)) and not re.search(r'\bright\b', lower)
    has_right = bool(re.search(r'\b(going\s+)?right\b', lower)) and not re.search(r'\bleft\b', lower)
    has_front = bool(re.search(r'\bin\s+front\b', lower)) and not re.search(r'\bbehind\b', lower)
    has_behind = bool(re.search(r'\bbehind\b', lower)) and not re.search(r'\bin\s+front\b', lower)

    for step in steps:
        if step.action != "extend_row":
            continue

        if has_left and step.direction == "right":
            logger.info("Auto-fix: flipping direction right -> left")
            step.direction = "left"
        elif has_right and step.direction == "left":
            logger.info("Auto-fix: flipping direction left -> right")
            step.direction = "right"
        elif has_front and step.direction in ("behind", "back"):
            logger.info("Auto-fix: flipping direction behind -> front")
            step.direction = "front"
        elif has_behind and step.direction in ("front", "in_front"):
            logger.info("Auto-fix: flipping direction front -> behind")
            step.direction = "behind"

    return steps


def auto_fix_each_end_caps(
    instruction: str, steps: List[BuildStep], start_grid: Grid
) -> List[BuildStep]:
    """Fix 'each end' cap positions after row extension.

    Bug: LLM caps at OLD endpoints after extending a row.
    Fix: Recompute endpoints and relocate cap steps.
    """
    lower = instruction.lower()
    if not re.search(r'\b(each|both)\s+(end|side)s?\b', lower):
        return steps

    # Find extend_row step
    extend_step = None
    extend_idx = -1
    for i, s in enumerate(steps):
        if s.action == "extend_row":
            extend_step = s
            extend_idx = i
            break

    if extend_step is None:
        return steps

    # Compute new row endpoints after extension
    ex = extend_step.position.get("x", 0)
    ez = extend_step.position.get("z", 0)
    count = extend_step.count if isinstance(extend_step.count, int) else 3
    offset = DIRECTION_OFFSETS.get(extend_step.direction, (100, 0))

    # Find existing row endpoints from start_grid
    color = extend_step.color.capitalize()
    existing_blocks = start_grid.find_color(color)
    if not existing_blocks:
        return steps

    # Get all x or z coords of existing blocks on the same axis
    if offset[0] != 0:  # horizontal row (x-axis)
        fixed_z = existing_blocks[0].z
        all_x = sorted(set(b.x for b in existing_blocks if b.z == fixed_z))
        # After extension
        new_end_x = ex + offset[0] * (count - 1)
        all_x_extended = sorted(set(all_x + [ex + offset[0] * i for i in range(count)]))
        new_min_x = min(all_x_extended)
        new_max_x = max(all_x_extended)

        # Fix subsequent stack/place steps at old endpoints
        for step in steps[extend_idx + 1:]:
            if step.action in ("stack", "place"):
                sx = step.position.get("x", None)
                sz = step.position.get("z", None)
                if sx is not None and sz == fixed_z:
                    old_min = min(all_x) if all_x else None
                    old_max = max(all_x) if all_x else None
                    if sx == old_max and new_max_x != old_max:
                        logger.info("Auto-fix each-end cap: x=%d -> x=%d", sx, new_max_x)
                        step.position["x"] = new_max_x
                    elif sx == old_min and new_min_x != old_min:
                        logger.info("Auto-fix each-end cap: x=%d -> x=%d", sx, new_min_x)
                        step.position["x"] = new_min_x

    elif offset[1] != 0:  # depth row (z-axis)
        fixed_x = existing_blocks[0].x
        all_z = sorted(set(b.z for b in existing_blocks if b.x == fixed_x))
        all_z_extended = sorted(set(all_z + [ez + offset[1] * i for i in range(count)]))
        new_min_z = min(all_z_extended)
        new_max_z = max(all_z_extended)

        for step in steps[extend_idx + 1:]:
            if step.action in ("stack", "place"):
                sx = step.position.get("x", None)
                sz = step.position.get("z", None)
                if sz is not None and sx == fixed_x:
                    old_min = min(all_z) if all_z else None
                    old_max = max(all_z) if all_z else None
                    if sz == old_max and new_max_z != old_max:
                        logger.info("Auto-fix each-end cap: z=%d -> z=%d", sz, new_max_z)
                        step.position["z"] = new_max_z
                    elif sz == old_min and new_min_z != old_min:
                        logger.info("Auto-fix each-end cap: z=%d -> z=%d", sz, new_min_z)
                        step.position["z"] = new_min_z

    return steps


def auto_fix_t_shape_extend(
    instruction: str, steps: List[BuildStep], start_grid: Grid
) -> List[BuildStep]:
    """Fix T-shape stem extension direction.

    Bug A: direction="on_top" or wrong axis (stacks vertically instead of extending)
    Bug B: action="stack" instead of "extend_row" at stem tip
    Fix: Analyze T-shape, compute correct direction away from junction.
    """
    lower = instruction.lower()
    if not re.search(r'\b(t[\s-]?shape|stem)\b', lower):
        return steps

    info = analyze_structure(start_grid)
    if not info.t_shapes:
        return steps

    t = info.t_shapes[0]
    junction = t["junction"]
    stem_tip = t["stem_tip"]
    stem_axis = t["stem_axis"]

    # Determine correct direction away from junction
    if stem_axis == "z":
        if stem_tip[1] < junction[1]:
            correct_dir = "behind"
        else:
            correct_dir = "front"
    else:
        if stem_tip[0] < junction[0]:
            correct_dir = "left"
        else:
            correct_dir = "right"

    for step in steps:
        if step.action == "extend_row":
            if step.direction in ("on_top", "up", "down"):
                logger.info("Auto-fix T-shape: direction '%s' -> '%s'", step.direction, correct_dir)
                step.direction = correct_dir
            # Fix start position to be beyond stem tip
            offset = DIRECTION_OFFSETS.get(correct_dir, (0, 0))
            expected_start_x = stem_tip[0] + offset[0]
            expected_start_z = stem_tip[1] + offset[1]
            if step.position.get("x") != expected_start_x or step.position.get("z") != expected_start_z:
                logger.info(
                    "Auto-fix T-shape start: (%s,%s) -> (%d,%d)",
                    step.position.get("x"), step.position.get("z"),
                    expected_start_x, expected_start_z,
                )
                step.position["x"] = expected_start_x
                step.position["z"] = expected_start_z

        elif step.action == "stack" and re.search(r'\bextend\b', lower):
            # Bug B: should be extend_row, not stack
            sx = step.position.get("x", 0)
            sz = step.position.get("z", 0)
            if (sx, sz) == stem_tip or (abs(sx - stem_tip[0]) <= 100 and abs(sz - stem_tip[1]) <= 100):
                logger.info("Auto-fix T-shape: converting stack to extend_row at stem tip")
                step.action = "extend_row"
                step.direction = correct_dir
                offset = DIRECTION_OFFSETS.get(correct_dir, (0, 0))
                step.position["x"] = stem_tip[0] + offset[0]
                step.position["z"] = stem_tip[1] + offset[1]

    # Fix cap positions at crossbar ends (if "each end" mentioned)
    if re.search(r'\b(each|both)\s+(end|side)s?\b', lower):
        crossbar_start = t["crossbar_start"]
        crossbar_end = t["crossbar_end"]
        cap_steps = [s for s in steps if s.action in ("stack", "place") and s != steps[0]]

        if len(cap_steps) >= 2:
            # Ensure caps are at the actual crossbar endpoints
            cap_steps[0].position["x"] = crossbar_start[0]
            cap_steps[0].position["z"] = crossbar_start[1]
            cap_steps[1].position["x"] = crossbar_end[0]
            cap_steps[1].position["z"] = crossbar_end[1]
            logger.info("Auto-fix T-shape caps: placed at crossbar endpoints (%d,%d) and (%d,%d)",
                        crossbar_start[0], crossbar_start[1], crossbar_end[0], crossbar_end[1])

    return steps
