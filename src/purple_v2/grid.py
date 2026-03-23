"""Grid model for 9x9 block-building game."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GridConfig:
    """Configuration for the 9x9 grid."""
    x_min: int = -400
    x_max: int = 400
    z_min: int = -400
    z_max: int = 400
    step: int = 100
    y_ground: int = 50
    y_step: int = 100
    max_height: int = 5  # max 5 blocks high

    @property
    def valid_xz(self) -> List[int]:
        return list(range(self.x_min, self.x_max + 1, self.step))

    @property
    def valid_y(self) -> List[int]:
        return [self.y_ground + i * self.y_step for i in range(self.max_height)]


VALID_COLORS = {
    "Red", "Blue", "Green", "Yellow", "Purple",
    "Orange", "White", "Black", "Brown", "Pink",
    "Grey", "Gray", "Cyan",
}

DIRECTION_OFFSETS: Dict[str, Tuple[int, int]] = {
    "right": (100, 0),
    "left": (-100, 0),
    "front": (0, 100),
    "in_front": (0, 100),
    "behind": (0, -100),
    "back": (0, -100),
}

CORNER_POSITIONS: Dict[str, Tuple[int, int]] = {
    "bottom_left": (-400, 400),
    "bottom_right": (400, 400),
    "top_left": (-400, -400),
    "top_right": (400, -400),
}


@dataclass(frozen=True)
class Block:
    """A single block on the grid."""
    color: str
    x: int
    y: int
    z: int

    def to_str(self) -> str:
        return f"{self.color},{self.x},{self.y},{self.z}"

    @classmethod
    def from_str(cls, s: str) -> "Block":
        parts = s.strip().split(",")
        if len(parts) != 4:
            raise ValueError(f"Invalid block string: {s}")
        color = parts[0].strip().capitalize()
        x, y, z = int(parts[1].strip()), int(parts[2].strip()), int(parts[3].strip())
        return cls(color=color, x=x, y=y, z=z)


class Grid:
    """Mutable grid containing blocks."""

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self.blocks: List[Block] = []

    def add_block(self, color: str, x: int, z: int, y: int | None = None) -> Block:
        """Add a block. If y is None, stack on top of existing blocks at (x, z)."""
        if y is None:
            y = self.next_y(x, z)
        block = Block(color=color.capitalize(), x=x, y=y, z=z)
        self.blocks.append(block)
        return block

    def next_y(self, x: int, z: int) -> int:
        """Get the next available y position at (x, z)."""
        blocks_at = [b for b in self.blocks if b.x == x and b.z == z]
        if not blocks_at:
            return self.config.y_ground
        max_y = max(b.y for b in blocks_at)
        return max_y + self.config.y_step

    def height_at(self, x: int, z: int) -> int:
        """Number of blocks stacked at (x, z)."""
        return sum(1 for b in self.blocks if b.x == x and b.z == z)

    def blocks_at(self, x: int, z: int) -> List[Block]:
        """All blocks at horizontal position (x, z), sorted by y."""
        return sorted(
            [b for b in self.blocks if b.x == x and b.z == z],
            key=lambda b: b.y,
        )

    def find_color(self, color: str) -> List[Block]:
        """Find all blocks of a specific color."""
        color = color.capitalize()
        return [b for b in self.blocks if b.color == color]

    def find_extreme(self, color: str, direction: str) -> Block | None:
        """Find the leftmost/rightmost/frontmost/backmost block of a color."""
        blocks = self.find_color(color)
        if not blocks:
            return None
        if direction == "leftmost":
            return min(blocks, key=lambda b: b.x)
        elif direction == "rightmost":
            return max(blocks, key=lambda b: b.x)
        elif direction == "frontmost":
            return max(blocks, key=lambda b: b.z)
        elif direction == "backmost":
            return min(blocks, key=lambda b: b.z)
        return None

    def most_common_color(self) -> str:
        """Return the most common color on the grid."""
        if not self.blocks:
            return "Purple"
        counts: Dict[str, int] = {}
        for b in self.blocks:
            counts[b.color] = counts.get(b.color, 0) + 1
        return max(counts, key=counts.get)  # type: ignore

    def to_str(self) -> str:
        """Serialize to semicolon-separated block strings."""
        return ";".join(b.to_str() for b in self.blocks)

    @classmethod
    def from_str(cls, s: str, config: GridConfig | None = None) -> "Grid":
        """Parse from semicolon-separated block strings."""
        grid = cls(config=config)
        if not s or not s.strip():
            return grid
        for part in s.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                block = Block.from_str(part)
                grid.blocks.append(block)
            except (ValueError, IndexError):
                continue
        return grid

    def copy(self) -> "Grid":
        """Create a deep copy."""
        g = Grid(config=self.config)
        g.blocks = list(self.blocks)
        return g

    def unique_positions(self) -> set[Tuple[int, int]]:
        """Set of unique (x, z) positions with blocks."""
        return {(b.x, b.z) for b in self.blocks}

    def unique_colors(self) -> set[str]:
        """Set of unique colors on the grid."""
        return {b.color for b in self.blocks}
