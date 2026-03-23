"""Tests for v3 spatial executor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v3.grid import Grid
from purple_v3.spatial_executor import SpatialExecutor
from purple_v3.build_planner import BuildStep
from purple_v3.response_formatter import format_build_response, validate_build_response


def test_stack_basic():
    grid = Grid()
    executor = SpatialExecutor(grid)
    executor.execute_plan([
        BuildStep(action="stack", color="Red", count=3, position={"x": 0, "z": 0})
    ])
    assert len(grid.blocks) == 3
    assert all(b.color == "Red" for b in grid.blocks)
    assert grid.blocks[0].y == 50
    assert grid.blocks[1].y == 150
    assert grid.blocks[2].y == 250


def test_extend_row():
    grid = Grid()
    executor = SpatialExecutor(grid)
    executor.execute_plan([
        BuildStep(action="extend_row", color="Green", count=4, position={"x": 0, "z": 0}, direction="right")
    ])
    assert len(grid.blocks) == 4
    xs = sorted(b.x for b in grid.blocks)
    assert xs == [0, 100, 200, 300]


def test_place_at_corners():
    grid = Grid()
    executor = SpatialExecutor(grid)
    executor.execute_plan([
        BuildStep(action="place_at_corners", color="Red", count=4)
    ])
    assert len(grid.blocks) == 4
    positions = {(b.x, b.z) for b in grid.blocks}
    assert (-400, 400) in positions
    assert (400, 400) in positions
    assert (-400, -400) in positions
    assert (400, -400) in positions


def test_trial_9_corners_with_stacking():
    """Trial 9: Place red in each corner, green on top of each."""
    grid = Grid()
    executor = SpatialExecutor(grid)
    executor.execute_plan([
        BuildStep(action="place", color="Red", count=1, position={"x": -400, "z": 400}),
        BuildStep(action="place", color="Red", count=1, position={"x": 400, "z": 400}),
        BuildStep(action="place", color="Red", count=1, position={"x": -400, "z": -400}),
        BuildStep(action="place", color="Red", count=1, position={"x": 400, "z": -400}),
        BuildStep(action="stack", color="Green", count=1, position={"x": -400, "z": 400}),
        BuildStep(action="stack", color="Green", count=1, position={"x": 400, "z": 400}),
        BuildStep(action="stack", color="Green", count=1, position={"x": -400, "z": -400}),
        BuildStep(action="stack", color="Green", count=1, position={"x": 400, "z": -400}),
    ])
    assert len(grid.blocks) == 8
    response = format_build_response(grid)
    is_valid, errors = validate_build_response(response)
    assert is_valid, errors

    # Check target structure
    target = "Red,-400,50,-400;Red,400,50,-400;Red,400,50,400;Red,-400,50,400;Green,-400,150,-400;Green,400,150,-400;Green,400,150,400;Green,-400,150,400"
    target_set = {s.strip() for s in target.split(";")}
    response_set = {s.strip() for s in response.replace("[BUILD];", "").split(";")}
    assert target_set == response_set


def test_trial_12_stacking():
    """Trial 12: Stack 3 red in bottom-right, then 2 yellow on top."""
    grid = Grid()
    executor = SpatialExecutor(grid)
    executor.execute_plan([
        BuildStep(action="stack", color="Red", count=3, position={"x": 400, "z": 400}),
        BuildStep(action="stack", color="Yellow", count=2, position={"x": 400, "z": 400}),
    ])
    assert len(grid.blocks) == 5
    assert grid.blocks[0].y == 50
    assert grid.blocks[4].y == 450
    response = format_build_response(grid)
    is_valid, _ = validate_build_response(response)
    assert is_valid


def test_response_format_valid():
    grid = Grid.from_str("Red,0,50,0;Blue,100,50,0")
    response = format_build_response(grid)
    assert response.startswith("[BUILD]")
    is_valid, errors = validate_build_response(response)
    assert is_valid, errors
