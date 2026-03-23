"""Tests for v3 structure analyzer and plan verifier."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v3.grid import Grid
from purple_v3.structure_analyzer import analyze_structure
from purple_v3.plan_verifier import verify_plan, auto_fix_direction, auto_fix_each_end_caps
from purple_v3.build_planner import BuildStep
from purple_v3.prompt_enricher import enrich_prompt


def test_detect_line():
    grid = Grid.from_str("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
    info = analyze_structure(grid)
    assert len(info.lines) == 1
    assert info.lines[0]["count"] == 3


def test_detect_stack():
    grid = Grid.from_str("Blue,0,50,0;Blue,0,150,0;Blue,0,250,0")
    info = analyze_structure(grid)
    assert len(info.stacks) == 1
    assert info.stacks[0]["height"] == 3


def test_detect_l_shape():
    grid = Grid.from_str("Purple,0,50,-100;Purple,0,50,0;Purple,0,50,100;Purple,100,50,100")
    info = analyze_structure(grid)
    assert len(info.l_shapes) == 1


def test_detect_t_shape():
    grid = Grid.from_str("Red,-100,50,0;Red,0,50,0;Red,100,50,0;Red,0,50,-100")
    info = analyze_structure(grid)
    assert len(info.t_shapes) == 1
    t = info.t_shapes[0]
    assert t["junction"] == (0, 0)


def test_auto_fix_direction():
    steps = [BuildStep(action="extend_row", color="Red", count=3, position={"x": 0, "z": 0}, direction="right")]
    fixed = auto_fix_direction("Build a row going left", steps)
    assert fixed[0].direction == "left"


def test_verify_plan_each_end():
    steps = [
        BuildStep(action="extend_row", color="Red", count=2, position={"x": 200, "z": 0}, direction="right"),
        BuildStep(action="stack", color="Red", count=1, position={"x": -100, "z": 0}),
    ]
    result = verify_plan("extend by 2 to the right, then stack on each end", steps, 3)
    assert result.has_critical is True  # only 1 stack step, needs 2 for "each end"


def test_enrichment_in_front():
    text = enrich_prompt("place a block in front of the red one")
    assert "REMINDER" in text
    assert "+z" in text


def test_enrichment_behind():
    text = enrich_prompt("place a block behind the stack")
    assert "REMINDER" in text
    assert "-z" in text


def test_enrichment_each_end():
    text = enrich_prompt("stack a block on each end")
    assert "REMINDER" in text
    assert "RECALCULATE" in text


def test_enrichment_no_match():
    text = enrich_prompt("do something simple")
    assert text == ""
