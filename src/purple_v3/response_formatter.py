"""Format and validate [BUILD] / [ASK] responses."""
from __future__ import annotations

from .grid import Grid, GridConfig, VALID_COLORS


def format_build_response(grid: Grid) -> str:
    """Format a grid as a [BUILD] response string."""
    if not grid.blocks:
        return "[BUILD]"
    parts = [b.to_str() for b in grid.blocks]
    return "[BUILD];" + ";".join(parts)


def validate_build_response(
    response: str, config: GridConfig | None = None
) -> tuple[bool, list[str]]:
    """Validate a [BUILD] response. Returns (is_valid, errors)."""
    config = config or GridConfig()
    errors: list[str] = []

    if not response.startswith("[BUILD]"):
        errors.append("Response must start with [BUILD]")
        return False, errors

    content = response[7:]  # after "[BUILD]"
    if not content or content == ";":
        return True, []  # empty build is technically valid

    if content.startswith(";"):
        content = content[1:]

    valid_xz = set(config.valid_xz)
    valid_y = set(config.valid_y)
    seen_positions: set[tuple[int, int, int]] = set()

    for part in content.split(";"):
        part = part.strip()
        if not part:
            continue

        pieces = part.split(",")
        if len(pieces) != 4:
            errors.append(f"Invalid block format: {part}")
            continue

        color = pieces[0].strip()
        if color.capitalize() not in VALID_COLORS:
            errors.append(f"Invalid color: {color}")

        try:
            x, y, z = int(pieces[1]), int(pieces[2]), int(pieces[3])
        except ValueError:
            errors.append(f"Non-integer coordinates: {part}")
            continue

        if x not in valid_xz:
            errors.append(f"Invalid x coordinate: {x}")
        if z not in valid_xz:
            errors.append(f"Invalid z coordinate: {z}")
        if y not in valid_y:
            errors.append(f"Invalid y coordinate: {y}")

        pos = (x, y, z)
        if pos in seen_positions:
            errors.append(f"Duplicate position: {pos}")
        seen_positions.add(pos)

    return len(errors) == 0, errors
