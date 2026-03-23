"""Tests for the spatial executor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v2.grid import Grid
from purple_v2.build_planner import BuildStep
from purple_v2.spatial_executor import SpatialExecutor


def test_stack():
    g = Grid()
    executor = SpatialExecutor(g)
    step = BuildStep(action="stack", color="Red", count=3, position={"x": 0, "z": 0})
    executor.execute_plan([step])
    assert len(g.blocks) == 3
    assert g.blocks[0].y == 50
    assert g.blocks[1].y == 150
    assert g.blocks[2].y == 250


def test_place():
    g = Grid()
    executor = SpatialExecutor(g)
    step = BuildStep(action="place", color="Blue", count=1, position={"x": 100, "z": 200})
    executor.execute_plan([step])
    assert len(g.blocks) == 1
    assert g.blocks[0].x == 100
    assert g.blocks[0].z == 200


def test_extend_row():
    g = Grid()
    executor = SpatialExecutor(g)
    step = BuildStep(
        action="extend_row", color="Green", count=4,
        position={"x": 0, "z": 0}, direction="right"
    )
    executor.execute_plan([step])
    assert len(g.blocks) == 4
    xs = sorted(b.x for b in g.blocks)
    assert xs == [0, 100, 200, 300]


def test_place_at_corners():
    g = Grid()
    executor = SpatialExecutor(g)
    step = BuildStep(action="place_at_corners", color="Yellow", count=4)
    executor.execute_plan([step])
    assert len(g.blocks) == 4
    positions = {(b.x, b.z) for b in g.blocks}
    assert (-400, 400) in positions
    assert (400, 400) in positions
    assert (-400, -400) in positions
    assert (400, -400) in positions


def test_extend_row_skips_existing():
    """extend_row at a position with same color should advance one step."""
    g = Grid.from_str("Red,0,50,0")
    executor = SpatialExecutor(g)
    step = BuildStep(
        action="extend_row", color="Red", count=2,
        position={"x": 0, "z": 0}, direction="right"
    )
    executor.execute_plan([step])
    # Should have original + 2 new, but starting from x=100
    assert len(g.blocks) == 3
    new_blocks = [b for b in g.blocks if b.x != 0]
    xs = sorted(b.x for b in new_blocks)
    assert xs == [100, 200]


def test_chain_reference_tracking():
    """Executor tracks last placed position per color."""
    g = Grid()
    executor = SpatialExecutor(g)
    steps = [
        BuildStep(action="place", color="Red", count=1, position={"x": 0, "z": 0}),
        BuildStep(action="place", color="Blue", count=1, position={"x": 100, "z": 0}),
    ]
    executor.execute_plan(steps)
    assert executor._last_position_by_color["Red"] == (0, 0)
    assert executor._last_position_by_color["Blue"] == (100, 0)
