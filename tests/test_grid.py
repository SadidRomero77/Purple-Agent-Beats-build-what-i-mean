"""Tests for the Grid model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v2.grid import Grid, GridConfig, Block


def test_add_block_ground():
    g = Grid()
    b = g.add_block("Red", 0, 0)
    assert b.y == 50
    assert b.color == "Red"


def test_stacking():
    g = Grid()
    g.add_block("Red", 0, 0)
    b2 = g.add_block("Blue", 0, 0)
    assert b2.y == 150


def test_from_str():
    g = Grid.from_str("Red,0,50,0;Blue,100,50,200")
    assert len(g.blocks) == 2
    assert g.blocks[0].color == "Red"
    assert g.blocks[1].x == 100


def test_to_str_roundtrip():
    original = "Red,0,50,0;Blue,100,150,200"
    g = Grid.from_str(original)
    result = g.to_str()
    assert "Red,0,50,0" in result
    assert "Blue,100,150,200" in result


def test_empty_grid():
    g = Grid.from_str("")
    assert len(g.blocks) == 0
    assert g.to_str() == ""


def test_height_at():
    g = Grid.from_str("Red,0,50,0;Red,0,150,0;Blue,100,50,0")
    assert g.height_at(0, 0) == 2
    assert g.height_at(100, 0) == 1
    assert g.height_at(200, 0) == 0


def test_find_color():
    g = Grid.from_str("Red,0,50,0;Blue,100,50,0;Red,200,50,0")
    reds = g.find_color("Red")
    assert len(reds) == 2


def test_find_extreme():
    g = Grid.from_str("Green,-200,50,0;Green,0,50,0;Green,300,50,0")
    left = g.find_extreme("Green", "leftmost")
    right = g.find_extreme("Green", "rightmost")
    assert left.x == -200
    assert right.x == 300
