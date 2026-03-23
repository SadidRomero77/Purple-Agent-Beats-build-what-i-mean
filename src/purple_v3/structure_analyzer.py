"""Analyze existing grid structures to detect shapes and provide context for the LLM planner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .grid import Block, Grid


@dataclass
class StructureInfo:
    """Analysis of existing structures on the grid."""
    lines: List[Dict] = field(default_factory=list)
    stacks: List[Dict] = field(default_factory=list)
    l_shapes: List[Dict] = field(default_factory=list)
    t_shapes: List[Dict] = field(default_factory=list)
    singles: List[Dict] = field(default_factory=list)
    color_extremes: Dict[str, Dict[str, Tuple[int, int]]] = field(default_factory=dict)

    def describe(self) -> str:
        """Generate natural language description for the LLM."""
        if not any([self.lines, self.stacks, self.l_shapes, self.t_shapes, self.singles]):
            return "The grid is empty."

        parts = []

        for t in self.t_shapes:
            parts.append(
                f"T-shape ({t['color']}): crossbar from ({t['crossbar_start'][0]},{t['crossbar_start'][1]}) "
                f"to ({t['crossbar_end'][0]},{t['crossbar_end'][1]}) along {'x' if t['crossbar_axis'] == 'x' else 'z'}-axis, "
                f"stem at ({t['stem_base'][0]},{t['stem_base'][1]}) extending {'along z' if t['stem_axis'] == 'z' else 'along x'} "
                f"to ({t['stem_tip'][0]},{t['stem_tip'][1]}), junction at ({t['junction'][0]},{t['junction'][1]})."
            )

        for l in self.l_shapes:
            parts.append(
                f"L-shape ({l['color']}): arm1 from ({l['arm1_start'][0]},{l['arm1_start'][1]}) "
                f"to ({l['corner'][0]},{l['corner'][1]}), "
                f"arm2 from ({l['corner'][0]},{l['corner'][1]}) "
                f"to ({l['arm2_end'][0]},{l['arm2_end'][1]})."
            )

        for line in self.lines:
            axis = "horizontal (x-axis)" if line["axis"] == "x" else "depth (z-axis)"
            parts.append(
                f"Line of {line['count']} {line['color']} blocks {axis} "
                f"from ({line['start'][0]},{line['start'][1]}) to ({line['end'][0]},{line['end'][1]})."
            )

        for stack in self.stacks:
            parts.append(
                f"Stack of {stack['height']} {stack['color']} blocks at ({stack['x']},{stack['z']})."
            )

        for single in self.singles:
            parts.append(
                f"Single {single['color']} block at ({single['x']},y={single['y']},{single['z']})."
            )

        if self.color_extremes:
            for color, extremes in self.color_extremes.items():
                extreme_parts = []
                for direction, pos in extremes.items():
                    extreme_parts.append(f"{direction} at ({pos[0]},{pos[1]})")
                parts.append(f"{color} extremes: {', '.join(extreme_parts)}.")

        return " ".join(parts)


def analyze_structure(grid: Grid) -> StructureInfo:
    """Analyze the grid to detect shapes and patterns."""
    info = StructureInfo()

    if not grid.blocks:
        return info

    # Group blocks by color
    by_color: Dict[str, List[Block]] = {}
    for b in grid.blocks:
        by_color.setdefault(b.color, []).append(b)

    # Analyze color extremes
    for color, blocks in by_color.items():
        ground_blocks = [b for b in blocks if b.y == grid.config.y_ground]
        if not ground_blocks:
            ground_blocks = blocks
        extremes: Dict[str, Tuple[int, int]] = {}
        if len(ground_blocks) > 1:
            leftmost = min(ground_blocks, key=lambda b: b.x)
            rightmost = max(ground_blocks, key=lambda b: b.x)
            frontmost = max(ground_blocks, key=lambda b: b.z)
            backmost = min(ground_blocks, key=lambda b: b.z)
            if leftmost.x != rightmost.x:
                extremes["leftmost"] = (leftmost.x, leftmost.z)
                extremes["rightmost"] = (rightmost.x, rightmost.z)
            if frontmost.z != backmost.z:
                extremes["frontmost"] = (frontmost.x, frontmost.z)
                extremes["backmost"] = (backmost.x, backmost.z)
        if extremes:
            info.color_extremes[color] = extremes

    # Group ground-level blocks by color for shape detection
    for color, blocks in by_color.items():
        ground_positions = sorted(
            {(b.x, b.z) for b in blocks if b.y == grid.config.y_ground}
        )

        if not ground_positions:
            # All blocks are stacked - treat as stacks
            pos_groups: Dict[Tuple[int, int], List[Block]] = {}
            for b in blocks:
                pos_groups.setdefault((b.x, b.z), []).append(b)
            for (x, z), blist in pos_groups.items():
                if len(blist) > 1:
                    info.stacks.append({"color": color, "x": x, "z": z, "height": len(blist)})
                else:
                    info.singles.append({"color": color, "x": blist[0].x, "y": blist[0].y, "z": blist[0].z})
            continue

        if len(ground_positions) == 1:
            x, z = ground_positions[0]
            height = sum(1 for b in blocks if b.x == x and b.z == z)
            if height > 1:
                info.stacks.append({"color": color, "x": x, "z": z, "height": height})
            else:
                b = blocks[0]
                info.singles.append({"color": color, "x": b.x, "y": b.y, "z": b.z})
            continue

        # Try to detect T-shape
        t_shape = _detect_t_shape(ground_positions, color)
        if t_shape:
            info.t_shapes.append(t_shape)
            # Add stack info for any blocks above ground at these positions
            _add_stacks(info, blocks, color, grid.config.y_ground)
            continue

        # Try to detect L-shape
        l_shape = _detect_l_shape(ground_positions, color)
        if l_shape:
            info.l_shapes.append(l_shape)
            _add_stacks(info, blocks, color, grid.config.y_ground)
            continue

        # Try to detect lines
        lines = _detect_lines(ground_positions, color)
        if lines:
            info.lines.extend(lines)
            _add_stacks(info, blocks, color, grid.config.y_ground)
            continue

        # Remaining: individual blocks
        for x, z in ground_positions:
            height = sum(1 for b in blocks if b.x == x and b.z == z)
            if height > 1:
                info.stacks.append({"color": color, "x": x, "z": z, "height": height})
            else:
                info.singles.append({"color": color, "x": x, "y": grid.config.y_ground, "z": z})

    return info


def _add_stacks(info: StructureInfo, blocks: List[Block], color: str, y_ground: int):
    """Add stack info for blocks above ground level."""
    pos_counts: Dict[Tuple[int, int], int] = {}
    for b in blocks:
        key = (b.x, b.z)
        pos_counts[key] = pos_counts.get(key, 0) + 1
    for (x, z), count in pos_counts.items():
        if count > 1:
            info.stacks.append({"color": color, "x": x, "z": z, "height": count})


def _detect_lines(positions: list, color: str) -> list:
    """Detect horizontal or depth lines."""
    lines = []

    # Check for x-axis line (same z)
    z_groups: Dict[int, list] = {}
    for x, z in positions:
        z_groups.setdefault(z, []).append(x)

    for z, xs in z_groups.items():
        xs.sort()
        if len(xs) >= 2:
            # Check if consecutive with step 100
            is_line = all(xs[i + 1] - xs[i] == 100 for i in range(len(xs) - 1))
            if is_line:
                lines.append({
                    "color": color, "axis": "x", "count": len(xs),
                    "start": (xs[0], z), "end": (xs[-1], z),
                })

    # Check for z-axis line (same x)
    x_groups: Dict[int, list] = {}
    for x, z in positions:
        x_groups.setdefault(x, []).append(z)

    for x, zs in x_groups.items():
        zs.sort()
        if len(zs) >= 2:
            is_line = all(zs[i + 1] - zs[i] == 100 for i in range(len(zs) - 1))
            if is_line:
                lines.append({
                    "color": color, "axis": "z", "count": len(zs),
                    "start": (x, zs[0]), "end": (x, zs[-1]),
                })

    return lines


def _detect_l_shape(positions: list, color: str) -> dict | None:
    """Detect an L-shape (two perpendicular arms meeting at a corner)."""
    if len(positions) < 3:
        return None

    # Group by x and z
    x_groups: Dict[int, list] = {}
    z_groups: Dict[int, list] = {}
    for x, z in positions:
        x_groups.setdefault(x, []).append(z)
        z_groups.setdefault(z, []).append(x)

    # L-shape: exactly 2 lines sharing a corner
    for x, zs in x_groups.items():
        if len(zs) >= 2:
            zs.sort()
            is_line = all(zs[i + 1] - zs[i] == 100 for i in range(len(zs) - 1))
            if not is_line:
                continue
            # Check if any endpoint connects to a perpendicular line
            for z_end in [zs[0], zs[-1]]:
                if z_end in z_groups:
                    xs = sorted(z_groups[z_end])
                    if len(xs) >= 2 and x in xs:
                        is_perp = all(xs[i + 1] - xs[i] == 100 for i in range(len(xs) - 1))
                        if is_perp:
                            # Found L-shape
                            arm1_other = zs[-1] if z_end == zs[0] else zs[0]
                            arm2_other = xs[-1] if x == xs[0] else xs[0]
                            return {
                                "color": color,
                                "corner": (x, z_end),
                                "arm1_start": (x, arm1_other),
                                "arm2_end": (arm2_other, z_end),
                            }
    return None


def _detect_t_shape(positions: list, color: str) -> dict | None:
    """Detect a T-shape (crossbar + stem with junction at interior point)."""
    if len(positions) < 4:
        return None

    pos_set = set(positions)

    # Group by x and z to find lines
    x_groups: Dict[int, list] = {}
    z_groups: Dict[int, list] = {}
    for x, z in positions:
        x_groups.setdefault(x, []).append(z)
        z_groups.setdefault(z, []).append(x)

    # Find longest line as potential crossbar
    lines = []
    for z, xs in z_groups.items():
        xs.sort()
        if len(xs) >= 3 and all(xs[i + 1] - xs[i] == 100 for i in range(len(xs) - 1)):
            lines.append(("x", z, xs))
    for x, zs in x_groups.items():
        zs.sort()
        if len(zs) >= 3 and all(zs[i + 1] - zs[i] == 100 for i in range(len(zs) - 1)):
            lines.append(("z", x, zs))

    lines.sort(key=lambda l: len(l[2]), reverse=True)

    for axis, fixed, coords in lines:
        # Check for a perpendicular stem at an interior point
        for i in range(1, len(coords) - 1):  # interior points only
            junction = coords[i]
            if axis == "x":
                jx, jz = junction, fixed
                # Stem goes along z from junction
                stem_zs = []
                for dz in [100, -100]:
                    sz = jz + dz
                    while (jx, sz) in pos_set and sz != jz:
                        stem_zs.append(sz)
                        sz += dz
                if stem_zs:
                    stem_zs.sort()
                    stem_tip_z = stem_zs[-1] if stem_zs[-1] != jz else stem_zs[0]
                    return {
                        "color": color,
                        "crossbar_axis": "x",
                        "crossbar_start": (coords[0], fixed),
                        "crossbar_end": (coords[-1], fixed),
                        "stem_axis": "z",
                        "stem_base": (jx, jz),
                        "stem_tip": (jx, stem_tip_z),
                        "junction": (jx, jz),
                    }
            else:
                jx, jz = fixed, junction
                stem_xs = []
                for dx in [100, -100]:
                    sx = jx + dx
                    while (sx, jz) in pos_set and sx != jx:
                        stem_xs.append(sx)
                        sx += dx
                if stem_xs:
                    stem_xs.sort()
                    stem_tip_x = stem_xs[-1] if stem_xs[-1] != jx else stem_xs[0]
                    return {
                        "color": color,
                        "crossbar_axis": "z",
                        "crossbar_start": (fixed, coords[0]),
                        "crossbar_end": (fixed, coords[-1]),
                        "stem_axis": "x",
                        "stem_base": (jx, jz),
                        "stem_tip": (stem_tip_x, jz),
                        "junction": (jx, jz),
                    }

    return None
