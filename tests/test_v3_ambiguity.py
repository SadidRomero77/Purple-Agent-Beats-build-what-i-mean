"""Tests for v4 precise regex ambiguity detection."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v3.ambiguity_detector import (
    detect_ambiguity,
    infer_color_from_context,
    infer_count_from_context,
    patch_instruction_with_color,
    patch_instruction_with_count,
)
from purple_v3.grid import Grid


# ── Color ambiguity tests ──

def test_color_missing_stack_blocks():
    """'stack four blocks in front of them' — no color between verb and noun."""
    instruction = "Stack five purple blocks in the middle, then stack four blocks in front of them."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_color is True


def test_color_present_stack_red_blocks():
    """'stack three red blocks' — color IS between verb and noun."""
    instruction = "Stack three red blocks in the bottom right corner."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_color is False


def test_color_missing_reference_not_counted():
    """'stack blocks behind the red one' — red is a reference, not the placed block's color."""
    instruction = "Stack three blocks behind the red one."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_color is True


def test_color_missing_place_block_on_top():
    """Trial 3a: 'place a block on top of the middle block' — no color."""
    instruction = "Place a blue block on the highlighted square. Place a blue block both to the left and to the right. Then place a block on top of the middle block."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_color is True


def test_color_missing_add_block():
    """Trial 4a: 'Add a block to the shorter side' — no color."""
    instruction = "Extend the longer side with two purple blocks. Add a block to the shorter side of the L."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_color is True


def test_fully_specified_no_ambiguity():
    """Fully specified instruction — no ambiguity at all."""
    instruction = "Stack four green blocks in the middle of the grid."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_color is False
    assert result.has_missing_count is False


# ── Count ambiguity tests ──

def test_count_missing_stack_red_blocks():
    """'stack red blocks directly in front' — no count."""
    instruction = "Stack two yellow blocks. Then stack red blocks directly in front."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_count is True


def test_count_missing_build_stack():
    """'Build a yellow stack' — no count for the stack."""
    instruction = "Build a yellow stack in front of the green one."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_count is True


def test_count_present_stack_three():
    """'Stack three red blocks' — count IS specified."""
    instruction = "Stack three red blocks in the corner."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_count is False


def test_count_present_place_a_block():
    """'Place a block' — 'a' implies count=1, not missing."""
    instruction = "Place a red block on top."
    result = detect_ambiguity(instruction, Grid())
    assert result.has_missing_count is False


# ── Color inference tests ──

def test_infer_single_color():
    color = infer_color_from_context("Stack five purple blocks, then stack blocks", Grid())
    assert color == "Purple"


def test_infer_last_color():
    color = infer_color_from_context("Stack red blocks, then build blue blocks, then add blocks", Grid())
    assert color == "Blue"


def test_infer_from_grid():
    grid = Grid.from_str("Red,0,50,0;Red,0,150,0;Blue,100,50,0")
    color = infer_color_from_context("stack blocks on top", grid)
    assert color == "Red"  # most common


def test_infer_no_context():
    color = infer_color_from_context("stack blocks", Grid())
    assert color == "Red"  # default, never Purple


# ── Count inference tests ──

def test_count_infer_from_adjacent_stack():
    grid = Grid.from_str("Red,0,50,0;Red,0,150,0;Red,0,250,0")
    count = infer_count_from_context("Build a blue stack to the right of the red one", grid)
    assert count == 3  # red stack is 3 high


def test_count_infer_default():
    count = infer_count_from_context("Build a stack somewhere", Grid())
    assert count == 3


# ── Patching tests ──

def test_patch_color():
    patched = patch_instruction_with_color("stack four blocks in front", "Yellow")
    assert "yellow" in patched.lower()


def test_patch_count():
    patched = patch_instruction_with_count("Build a yellow stack in front", 3)
    assert "3" in patched
