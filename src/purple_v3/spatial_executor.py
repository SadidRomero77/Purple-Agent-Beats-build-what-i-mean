"""Deterministic spatial executor: converts build steps into grid blocks."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .build_planner import BuildStep
from .grid import Block, Grid, GridConfig, DIRECTION_OFFSETS, CORNER_POSITIONS

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    pass


class SpatialExecutor:
    """Execute build steps deterministically on a grid."""

    def __init__(self, grid: Grid):
        self.grid = grid
        # Track last placed position per color for chain references
        self._last_position_by_color: Dict[str, Tuple[int, int]] = {}
        # Initialize from existing blocks
        for b in grid.blocks:
            self._last_position_by_color[b.color] = (b.x, b.z)

    def execute_plan(self, steps: List[BuildStep]) -> None:
        """Execute all steps in order."""
        for i, step in enumerate(steps):
            try:
                self._execute_step(step)
            except Exception as exc:
                logger.warning("Step %d failed: %s (step=%s)", i + 1, exc, step)
                raise ExecutionError(f"Step {i + 1} failed: {exc}") from exc

    def _execute_step(self, step: BuildStep) -> None:
        """Execute a single build step."""
        handler = {
            "stack": self._exec_stack,
            "place": self._exec_place,
            "place_relative": self._exec_place_relative,
            "extend_row": self._exec_extend_row,
            "place_at_corners": self._exec_place_at_corners,
        }.get(step.action)

        if handler is None:
            raise ExecutionError(f"Unknown action: {step.action}")
        handler(step)

    def _resolve_color(self, color: str) -> str:
        """Resolve Uncolored to a sensible default."""
        if color.lower() in ("uncolored", "unknown", "unspecified", "?"):
            return self.grid.most_common_color()
        return color.capitalize()

    def _resolve_count(self, count, step: BuildStep) -> int:
        """Resolve Uncounted to a sensible default."""
        if isinstance(count, str) and count.lower() in ("uncounted", "unknown", "unspecified", "?"):
            # Try to match height of nearby stack
            x = step.position.get("x", 0)
            z = step.position.get("z", 0)
            for dx, dz in [(100, 0), (-100, 0), (0, 100), (0, -100)]:
                h = self.grid.height_at(x + dx, z + dz)
                if h > 0:
                    return h
            return 3  # default
        if isinstance(count, str):
            try:
                return int(count)
            except ValueError:
                return 3
        return int(count)

    def _resolve_position(self, step: BuildStep) -> Tuple[int, int]:
        """Resolve position from step, handling references."""
        pos = step.position

        # Direct position
        if "x" in pos and "z" in pos:
            return int(pos["x"]), int(pos["z"])

        # Reference-based position
        if step.reference:
            ref_pos = self._resolve_reference(step.reference)
            if ref_pos and step.direction:
                offset = DIRECTION_OFFSETS.get(step.direction, (0, 0))
                return ref_pos[0] + offset[0], ref_pos[1] + offset[1]
            if ref_pos:
                return ref_pos

        raise ExecutionError(f"Cannot resolve position for step: {step}")

    def _resolve_reference(self, reference: str) -> Tuple[int, int] | None:
        """Resolve a reference string to a position."""
        ref_lower = reference.lower()

        # Check for color references like "Red_block_at_0_0"
        for color in self._last_position_by_color:
            if color.lower() in ref_lower:
                return self._last_position_by_color[color]

        # Check for position in reference like "block_at_100_200"
        import re
        m = re.search(r'at[_\s]+(-?\d+)[_\s]+(-?\d+)', reference)
        if m:
            return int(m.group(1)), int(m.group(2))

        # Check existing grid blocks by color
        for color, pos in self._last_position_by_color.items():
            if color.lower() in ref_lower:
                return pos

        return None

    def _exec_stack(self, step: BuildStep) -> None:
        """Stack N blocks vertically at (x, z)."""
        x, z = self._resolve_position(step)
        color = self._resolve_color(step.color)
        count = self._resolve_count(step.count, step)

        for _ in range(count):
            self.grid.add_block(color, x, z)

        self._last_position_by_color[color] = (x, z)
        logger.debug("Stacked %d %s at (%d,%d)", count, color, x, z)

    def _exec_place(self, step: BuildStep) -> None:
        """Place a single block at absolute position."""
        x, z = self._resolve_position(step)
        color = self._resolve_color(step.color)

        self.grid.add_block(color, x, z)
        self._last_position_by_color[color] = (x, z)
        logger.debug("Placed %s at (%d,%d)", color, x, z)

    def _exec_place_relative(self, step: BuildStep) -> None:
        """Place a block relative to a reference."""
        # Resolve reference position
        ref_pos = None
        if step.reference:
            ref_pos = self._resolve_reference(step.reference)

        if ref_pos is None:
            # Fallback to position if given
            if "x" in step.position and "z" in step.position:
                ref_pos = (int(step.position["x"]), int(step.position["z"]))
            else:
                raise ExecutionError(f"Cannot resolve reference: {step.reference}")

        color = self._resolve_color(step.color)
        offset = DIRECTION_OFFSETS.get(step.direction, (0, 0))
        x = ref_pos[0] + offset[0]
        z = ref_pos[1] + offset[1]

        self.grid.add_block(color, x, z)
        self._last_position_by_color[color] = (x, z)
        logger.debug("Placed %s relative at (%d,%d)", color, x, z)

    def _exec_extend_row(self, step: BuildStep) -> None:
        """Place N blocks in a line from start position in a direction.

        Same-color skip-forward rule (two cases):
        - Case 1: relative_to in position → ALWAYS advance start one step
        - Case 2: absolute coords → advance start only if same-color block exists at start
        Then place count blocks contiguously from the (possibly advanced) start.
        """
        pos = step.position
        color = self._resolve_color(step.color)
        count = self._resolve_count(step.count, step)
        direction = step.direction

        offset = DIRECTION_OFFSETS.get(direction, (100, 0))
        dx, dz = offset

        # Resolve base start position
        if "x" in pos and "z" in pos:
            start_x, start_z = int(pos["x"]), int(pos["z"])
        elif step.reference:
            ref = self._resolve_reference(step.reference)
            if ref is None:
                raise ExecutionError(f"Cannot resolve reference: {step.reference}")
            start_x, start_z = ref
        else:
            raise ExecutionError(f"Cannot resolve position for extend_row: {step}")

        # Same-color skip-forward rule
        if "relative_to" in pos:
            # Case 1: relative reference → ALWAYS advance one step past reference
            logger.info("extend_row: relative_to reference, advancing start one step %s", direction)
            start_x += dx
            start_z += dz
        else:
            # Case 2: absolute coords → advance only if same-color block at start position
            existing = self.grid.blocks_at(start_x, start_z)
            if existing and any(b.color == color for b in existing):
                logger.info(
                    "extend_row: start (%d,%d) already has %s, advancing one step %s",
                    start_x, start_z, color, direction,
                )
                start_x += dx
                start_z += dz

        # Place count blocks contiguously from start
        last_x, last_z = start_x, start_z
        for i in range(count):
            bx = start_x + dx * i
            bz = start_z + dz * i
            # Validate bounds
            if bx < self.grid.config.x_min or bx > self.grid.config.x_max:
                break
            if bz < self.grid.config.z_min or bz > self.grid.config.z_max:
                break
            self.grid.add_block(color, bx, bz)
            last_x, last_z = bx, bz

        self._last_position_by_color[color] = (last_x, last_z)
        logger.debug("Extended row of %d %s from (%d,%d) %s", count, color, start_x, start_z, direction)

    def _exec_place_at_corners(self, step: BuildStep) -> None:
        """Place blocks at grid corners."""
        color = self._resolve_color(step.color)
        for name, (cx, cz) in CORNER_POSITIONS.items():
            self.grid.add_block(color, cx, cz)
            self._last_position_by_color[color] = (cx, cz)
        logger.debug("Placed %s at all corners", color)
